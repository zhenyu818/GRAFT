// Package controller implements a persistent, single-owner recovery state machine.
package controller

import (
	"context"
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"reflect"
	"sort"
	"time"

	"github.com/prometheus/client_golang/prometheus"
	training "github.com/zhenyu818/GRAFT/api/v1alpha1"
	corev1 "k8s.io/api/core/v1"
	apierrors "k8s.io/apimachinery/pkg/api/errors"
	"k8s.io/apimachinery/pkg/api/meta"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/apis/meta/v1/unstructured"
	"k8s.io/apimachinery/pkg/types"
	ctrl "sigs.k8s.io/controller-runtime"
	"sigs.k8s.io/controller-runtime/pkg/client"
	"sigs.k8s.io/controller-runtime/pkg/controller"
	"sigs.k8s.io/controller-runtime/pkg/controller/controllerutil"
	"sigs.k8s.io/controller-runtime/pkg/metrics"
)

const finalizer = "training.graft.dev/cleanup"

var recoveries = prometheus.NewCounterVec(prometheus.CounterOpts{
	Name: "graft_recoveries_total", Help: "Coordinated recovery decisions persisted to TrainingRun status.",
}, []string{"reason"})

func init() { metrics.Registry.MustRegister(recoveries) }

type Reconciler struct {
	client.Client
	ArtifactRoot string
	Concurrency  int
	Now          func() time.Time
}

type progress struct {
	Attempt int `json:"attempt"`
	Step    int `json:"step"`
}

func (r *Reconciler) clock() metav1.Time {
	if r.Now != nil {
		return metav1.NewTime(r.Now())
	}
	return metav1.Now()
}

func (r *Reconciler) SetupWithManager(mgr ctrl.Manager) error {
	job := &unstructured.Unstructured{}
	job.SetGroupVersionKind(VolcanoGVK)
	return ctrl.NewControllerManagedBy(mgr).Named("trainingrun").
		For(&training.TrainingRun{}).Owns(job).
		WithOptions(controller.Options{MaxConcurrentReconciles: r.Concurrency}).Complete(r)
}

func (r *Reconciler) status(ctx context.Context, run *training.TrainingRun, phase, reason, message string) error {
	run.Status.Phase = phase
	run.Status.ObservedGeneration = run.Generation
	ready := metav1.ConditionFalse
	if phase == "Succeeded" {
		ready = metav1.ConditionTrue
	}
	meta.SetStatusCondition(&run.Status.Conditions, metav1.Condition{
		Type: "Ready", Status: ready, Reason: reason, Message: message, ObservedGeneration: run.Generation,
	})
	return r.Status().Update(ctx, run)
}

func (r *Reconciler) Reconcile(ctx context.Context, req ctrl.Request) (ctrl.Result, error) {
	run := new(training.TrainingRun)
	if err := r.Get(ctx, req.NamespacedName, run); err != nil {
		return ctrl.Result{}, client.IgnoreNotFound(err)
	}
	if !run.DeletionTimestamp.IsZero() {
		if controllerutil.ContainsFinalizer(run, finalizer) {
			gone, err := r.cleanup(ctx, run)
			if err != nil {
				return ctrl.Result{}, err
			}
			if !gone {
				return ctrl.Result{RequeueAfter: time.Second}, nil
			}
			controllerutil.RemoveFinalizer(run, finalizer)
			return ctrl.Result{}, r.Update(ctx, run)
		}
		return ctrl.Result{}, nil
	}
	if !controllerutil.ContainsFinalizer(run, finalizer) {
		controllerutil.AddFinalizer(run, finalizer)
		return ctrl.Result{Requeue: true}, r.Update(ctx, run)
	}
	if run.Status.Phase == "Succeeded" || run.Status.Phase == "Failed" {
		return ctrl.Result{}, nil
	}
	now := r.clock()
	if run.Status.SubmittedAt == nil {
		run.Status.SubmittedAt = &now
	}
	if run.Spec.Suspend && run.Status.JobName == "" {
		if run.Status.Phase != "Suspended" {
			return ctrl.Result{}, r.status(ctx, run, "Suspended", "AdmissionHeld", "Admission is held by the submitted configuration.")
		}
		return ctrl.Result{}, nil
	}
	if run.Status.Phase == "Recovering" {
		gone, err := r.cleanup(ctx, run)
		if err != nil {
			return ctrl.Result{}, err
		}
		if !gone || (run.Status.NextRetryAt != nil && now.Before(run.Status.NextRetryAt)) {
			return ctrl.Result{RequeueAfter: time.Second}, nil
		}
		run.Status.Attempt++
		run.Status.AttemptStartedAt = nil
		run.Status.LastProgressAt = nil
		run.Status.LastStep = 0
		run.Status.WorkerUIDs = nil
		run.Status.NextRetryAt = nil
		return ctrl.Result{Requeue: true}, r.status(ctx, run, "Queued", "RecoveryAdmitted", "The previous worker group has stopped; the next attempt is admitted.")
	}
	job := &unstructured.Unstructured{}
	job.SetGroupVersionKind(VolcanoGVK)
	name := AttemptName(run)
	err := r.Get(ctx, types.NamespacedName{Namespace: run.Namespace, Name: name}, job)
	if apierrors.IsNotFound(err) {
		if run.Status.Phase == "Running" {
			return r.recover(ctx, run, "WorkerGroupRemoved")
		}
		cm, e := ConfigMap(run)
		if e != nil {
			return ctrl.Result{}, e
		}
		for _, obj := range []client.Object{cm, MasterService(run), VolcanoJob(run)} {
			if e := r.Create(ctx, obj); e != nil && !apierrors.IsAlreadyExists(e) {
				return ctrl.Result{}, e
			}
		}
		run.Status.JobName = name
		return ctrl.Result{RequeueAfter: time.Second}, r.status(ctx, run, "Queued", "SubmittedToScheduler", "The worker group is waiting for gang admission.")
	}
	if err != nil {
		return ctrl.Result{}, err
	}
	pods := &corev1.PodList{}
	if err := r.List(ctx, pods, client.InNamespace(run.Namespace), client.MatchingLabels{AttemptLabel: name}); err != nil {
		return ctrl.Result{}, err
	}
	jobPhase, _, _ := unstructured.NestedString(job.Object, "status", "state", "phase")
	for _, p := range pods.Items {
		if p.Status.Phase == corev1.PodFailed {
			return r.recover(ctx, run, "WorkerFailed")
		}
	}
	switch jobPhase {
	case "Failed", "Aborting", "Aborted", "Terminated":
		return r.recover(ctx, run, "WorkerGroupFailed")
	}
	allRunning := len(pods.Items) == int(run.Spec.Replicas)
	allSucceeded := allRunning
	uids := make([]string, 0, len(pods.Items))
	for _, p := range pods.Items {
		allRunning = allRunning && p.Status.Phase == corev1.PodRunning
		allSucceeded = allSucceeded && p.Status.Phase == corev1.PodSucceeded
		uids = append(uids, string(p.UID))
	}
	sort.Strings(uids)
	dir := filepath.Join(r.ArtifactRoot, string(run.UID), "attempts", fmt.Sprint(run.Status.Attempt))
	var done progress
	if readJSON(filepath.Join(dir, "complete.json"), &done) == nil && done.Attempt == run.Status.Attempt && (allSucceeded || jobPhase == "Completed") {
		run.Status.FinishedAt = &now
		run.Status.LastStep = done.Step
		return ctrl.Result{}, r.status(ctx, run, "Succeeded", "TrainingCompleted", "All workers completed and published the final training result.")
	}
	changed := run.Status.JobName != name
	run.Status.JobName = name
	if allRunning && run.Status.AttemptStartedAt == nil {
		run.Status.AttemptStartedAt = &now
		if run.Status.StartedAt == nil {
			run.Status.StartedAt = &now
		}
		run.Status.WorkerUIDs = uids
		changed = true
	}
	if run.Status.AttemptStartedAt != nil && !reflect.DeepEqual(run.Status.WorkerUIDs, uids) {
		return r.recover(ctx, run, "WorkerReplaced")
	}
	var current progress
	if readJSON(filepath.Join(dir, "progress.json"), &current) == nil && current.Attempt == run.Status.Attempt {
		if run.Status.LastProgressAt == nil || current.Step > run.Status.LastStep {
			run.Status.LastStep = current.Step
			run.Status.LastProgressAt = &now
			changed = true
		}
	}
	if run.Status.AttemptStartedAt != nil {
		last := run.Status.AttemptStartedAt
		timeout := run.Spec.StartupTimeoutSeconds
		if run.Status.LastProgressAt != nil {
			last = run.Status.LastProgressAt
			timeout = run.Spec.ProgressTimeoutSeconds
		}
		if now.Sub(last.Time) > time.Duration(timeout)*time.Second {
			return r.recover(ctx, run, "ProgressTimeout")
		}
	}
	phase := "Queued"
	if run.Status.AttemptStartedAt != nil {
		phase = "Running"
	}
	if changed || run.Status.Phase != phase {
		return ctrl.Result{RequeueAfter: time.Second}, r.status(ctx, run, phase, "ProgressObserved", "The active worker group is being supervised.")
	}
	return ctrl.Result{RequeueAfter: time.Second}, nil
}

func (r *Reconciler) recover(ctx context.Context, run *training.TrainingRun, reason string) (ctrl.Result, error) {
	now := r.clock()
	run.Status.Recoveries = append(run.Status.Recoveries, training.Recovery{Attempt: run.Status.Attempt, Reason: reason, DetectedAt: now, DetectedAtUnixMs: now.UnixMilli(), LastStep: run.Status.LastStep})
	if run.Status.Attempt >= run.Spec.MaxRetries {
		run.Status.FinishedAt = &now
		_, err := r.cleanup(ctx, run)
		if err != nil {
			return ctrl.Result{}, err
		}
		return ctrl.Result{}, r.status(ctx, run, "Failed", "RetryBudgetConsumed", "The configured recovery budget has been consumed.")
	}
	delay := time.Second * time.Duration(1<<min(run.Status.Attempt, 3))
	next := metav1.NewTime(now.Add(delay))
	run.Status.NextRetryAt = &next
	err := r.status(ctx, run, "Recovering", reason, "Stopping the entire worker group before restoring a committed checkpoint.")
	if err == nil {
		recoveries.WithLabelValues(reason).Inc()
	}
	return ctrl.Result{RequeueAfter: time.Second}, err
}

// cleanup waits for old workers to disappear before releasing the next attempt.
// Artifact files are retained as the durable experiment and recovery record.
func (r *Reconciler) cleanup(ctx context.Context, run *training.TrainingRun) (bool, error) {
	jobs := &unstructured.UnstructuredList{}
	jobs.SetGroupVersionKind(VolcanoGVK.GroupVersion().WithKind("JobList"))
	if err := r.List(ctx, jobs, client.InNamespace(run.Namespace), client.MatchingLabels{RunLabel: string(run.UID)}); err != nil {
		return false, err
	}
	for i := range jobs.Items {
		if err := r.Delete(ctx, &jobs.Items[i], client.PropagationPolicy(metav1.DeletePropagationForeground)); err != nil && !apierrors.IsNotFound(err) {
			return false, err
		}
	}
	pods := &corev1.PodList{}
	if err := r.List(ctx, pods, client.InNamespace(run.Namespace), client.MatchingLabels{RunLabel: string(run.UID)}); err != nil {
		return false, err
	}
	for i := range pods.Items {
		if err := r.Delete(ctx, &pods.Items[i], client.GracePeriodSeconds(0)); err != nil && !apierrors.IsNotFound(err) {
			return false, err
		}
	}
	if len(jobs.Items) > 0 || len(pods.Items) > 0 {
		return false, nil
	}
	for _, obj := range []client.Object{&corev1.ConfigMap{ObjectMeta: metav1.ObjectMeta{Name: AttemptName(run), Namespace: run.Namespace}}, &corev1.Service{ObjectMeta: metav1.ObjectMeta{Name: AttemptName(run), Namespace: run.Namespace}}} {
		if err := r.Delete(ctx, obj); err != nil && !apierrors.IsNotFound(err) {
			return false, err
		}
	}
	return true, nil
}

func readJSON(path string, target interface{}) error {
	b, err := os.ReadFile(path)
	if err != nil {
		return err
	}
	return json.Unmarshal(b, target)
}
