package controller

import (
	"context"
	"os"
	"path/filepath"
	"testing"
	"time"

	training "github.com/zhenyu818/GRAFT/api/v1alpha1"
	corev1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/apis/meta/v1/unstructured"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/types"
	ctrl "sigs.k8s.io/controller-runtime"
	"sigs.k8s.io/controller-runtime/pkg/client/fake"
)

func fixture(t *testing.T) (*Reconciler, ctrl.Request) {
	t.Helper()
	scheme := runtime.NewScheme()
	_ = corev1.AddToScheme(scheme)
	_ = training.AddToScheme(scheme)
	run := &training.TrainingRun{TypeMeta: metav1.TypeMeta{APIVersion: training.GroupVersion.String(), Kind: "TrainingRun"},
		ObjectMeta: metav1.ObjectMeta{Name: "training", Namespace: "graft-system", UID: "12345678-1234-5678-9012-123456789012", Generation: 1},
		Spec:       training.TrainingRunSpec{Workload: "ddp", Image: "graft-worker:0.1.0", Replicas: 2, GPUsPerWorker: 1, Queue: "graft", PriorityClassName: "graft-normal", Steps: 40, MaxRetries: 2, ProgressTimeoutSeconds: 10, StartupTimeoutSeconds: 60},
	}
	c := fake.NewClientBuilder().WithScheme(scheme).WithStatusSubresource(&training.TrainingRun{}).WithObjects(run).Build()
	r := &Reconciler{Client: c, ArtifactRoot: t.TempDir(), Concurrency: 4, Now: func() time.Time { return time.Unix(1000, 0) }}
	return r, ctrl.Request{NamespacedName: types.NamespacedName{Name: run.Name, Namespace: run.Namespace}}
}

func reconcile(t *testing.T, r *Reconciler, req ctrl.Request) {
	t.Helper()
	if _, err := r.Reconcile(context.Background(), req); err != nil {
		t.Fatal(err)
	}
}
func current(t *testing.T, r *Reconciler, req ctrl.Request) *training.TrainingRun {
	t.Helper()
	run := new(training.TrainingRun)
	if err := r.Get(context.Background(), req.NamespacedName, run); err != nil {
		t.Fatal(err)
	}
	return run
}
func jobs(t *testing.T, r *Reconciler) []unstructured.Unstructured {
	t.Helper()
	list := &unstructured.UnstructuredList{}
	list.SetGroupVersionKind(VolcanoGVK.GroupVersion().WithKind("JobList"))
	if err := r.List(context.Background(), list); err != nil {
		t.Fatal(err)
	}
	return list.Items
}

func TestRepeatedEventsCreateOneGang(t *testing.T) {
	r, req := fixture(t)
	for i := 0; i < 10; i++ {
		reconcile(t, r, req)
	}
	all := jobs(t, r)
	if len(all) != 1 {
		t.Fatalf("expected one worker group, got %d", len(all))
	}
	minimum, _, _ := unstructured.NestedInt64(all[0].Object, "spec", "minAvailable")
	if minimum != 2 {
		t.Fatalf("two training ranks must be admitted together: %d", minimum)
	}
	tasks, _, _ := unstructured.NestedSlice(all[0].Object, "spec", "tasks")
	for _, item := range tasks {
		task := item.(map[string]interface{})
		containers, _, _ := unstructured.NestedSlice(task, "template", "spec", "containers")
		gpu, _, _ := unstructured.NestedInt64(containers[0].(map[string]interface{}), "resources", "limits", "nvidia.com/gpu")
		if gpu != 1 {
			t.Fatalf("each rank must request one GPU: %d", gpu)
		}
	}
}

func TestRecoveryFencesThePreviousWorkerGroup(t *testing.T) {
	r, req := fixture(t)
	reconcile(t, r, req)
	reconcile(t, r, req)
	run := current(t, r, req)
	old := AttemptName(run)
	for index, phase := range []corev1.PodPhase{corev1.PodFailed, corev1.PodRunning} {
		p := &corev1.Pod{ObjectMeta: metav1.ObjectMeta{Name: old + string(rune('a'+index)), Namespace: run.Namespace,
			Labels: map[string]string{RunLabel: string(run.UID), AttemptLabel: old}}, Status: corev1.PodStatus{Phase: phase}}
		if err := r.Create(context.Background(), p); err != nil {
			t.Fatal(err)
		}
	}
	reconcile(t, r, req)
	if got := current(t, r, req); got.Status.Phase != "Recovering" || got.Status.Attempt != 0 {
		t.Fatalf("unexpected recovery decision: %+v", got.Status)
	}
	reconcile(t, r, req)
	if len(jobs(t, r)) != 0 {
		t.Fatal("the old group must be deleted before replacement")
	}
	r.Now = func() time.Time { return time.Unix(1010, 0) }
	reconcile(t, r, req)
	reconcile(t, r, req)
	got := current(t, r, req)
	if got.Status.Attempt != 1 || len(got.Status.Recoveries) != 1 {
		t.Fatalf("unexpected attempt accounting: %+v", got.Status)
	}
	all := jobs(t, r)
	if len(all) != 1 || all[0].GetName() == old {
		t.Fatal("the replacement must use a new deterministic attempt identity")
	}
}

func TestQueuedJobsUseAdmissionRatherThanProgressTimeout(t *testing.T) {
	r, req := fixture(t)
	reconcile(t, r, req)
	reconcile(t, r, req)
	r.Now = func() time.Time { return time.Unix(4000, 0) }
	reconcile(t, r, req)
	got := current(t, r, req)
	if got.Status.Phase != "Queued" || len(got.Status.Recoveries) > 0 {
		t.Fatalf("queue waiting must preserve the retry budget: %+v", got.Status)
	}
}

func TestCleanupRetainsDurableArtifacts(t *testing.T) {
	r, req := fixture(t)
	reconcile(t, r, req)
	reconcile(t, r, req)
	run := current(t, r, req)
	directory := filepath.Join(r.ArtifactRoot, string(run.UID))
	if err := os.MkdirAll(directory, 0700); err != nil {
		t.Fatal(err)
	}
	path := filepath.Join(directory, "checkpoint-record")
	if err := os.WriteFile(path, []byte("retained"), 0600); err != nil {
		t.Fatal(err)
	}
	if err := r.Delete(context.Background(), run); err != nil {
		t.Fatal(err)
	}
	for i := 0; i < 3; i++ {
		reconcile(t, r, req)
	}
	if _, err := os.Stat(path); err != nil {
		t.Fatal("durable training artifacts must survive resource cleanup", err)
	}
}

func TestRetryBudgetStopsReplacement(t *testing.T) {
	r, req := fixture(t)
	run := current(t, r, req)
	run.Spec.MaxRetries = 0
	if err := r.Update(context.Background(), run); err != nil {
		t.Fatal(err)
	}
	reconcile(t, r, req)
	reconcile(t, r, req)
	run = current(t, r, req)
	p := &corev1.Pod{ObjectMeta: metav1.ObjectMeta{Name: "failed", Namespace: run.Namespace, Labels: map[string]string{RunLabel: string(run.UID), AttemptLabel: AttemptName(run)}}, Status: corev1.PodStatus{Phase: corev1.PodFailed}}
	if err := r.Create(context.Background(), p); err != nil {
		t.Fatal(err)
	}
	reconcile(t, r, req)
	reconcile(t, r, req)
	if got := current(t, r, req); got.Status.Phase != "Failed" || got.Status.Attempt != 0 {
		t.Fatalf("unexpected budget outcome: %+v", got.Status)
	}
	if len(jobs(t, r)) != 0 {
		t.Fatal("a consumed budget must stop the worker group")
	}
}

func TestExistingChildIsAdoptedAfterAnInterruptedStatusWrite(t *testing.T) {
	r, req := fixture(t)
	run := current(t, r, req)
	if err := r.Create(context.Background(), VolcanoJob(run)); err != nil {
		t.Fatal(err)
	}
	reconcile(t, r, req)
	reconcile(t, r, req)
	got := current(t, r, req)
	if got.Status.JobName != AttemptName(got) || got.Status.Phase != "Queued" {
		t.Fatalf("the existing attempt must be recorded: %+v", got.Status)
	}
	if len(jobs(t, r)) != 1 {
		t.Fatal("an interrupted status write must preserve one worker group")
	}
}

func TestAdmissionHoldCanBeReleased(t *testing.T) {
	r, req := fixture(t)
	run := current(t, r, req)
	run.Spec.Suspend = true
	if err := r.Update(context.Background(), run); err != nil {
		t.Fatal(err)
	}
	reconcile(t, r, req)
	reconcile(t, r, req)
	if got := current(t, r, req); got.Status.Phase != "Suspended" || len(jobs(t, r)) != 0 {
		t.Fatal("held admission must preserve resources without creating workers")
	}
	run = current(t, r, req)
	run.Spec.Suspend = false
	run.Generation = 2
	if err := r.Update(context.Background(), run); err != nil {
		t.Fatal(err)
	}
	reconcile(t, r, req)
	if got := current(t, r, req); got.Status.Phase != "Queued" || got.Status.ObservedGeneration != 2 || len(jobs(t, r)) != 1 {
		t.Fatalf("released admission must create one current-generation attempt: %+v", got.Status)
	}
}
