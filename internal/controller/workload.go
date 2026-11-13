package controller

import (
	"encoding/json"
	"fmt"
	"strconv"

	training "github.com/zhenyu818/GRAFT/api/v1alpha1"
	corev1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/apis/meta/v1/unstructured"
	"k8s.io/apimachinery/pkg/runtime/schema"
)

var VolcanoGVK = schema.GroupVersionKind{Group: "batch.volcano.sh", Version: "v1alpha1", Kind: "Job"}

const RunLabel = "training.graft.dev/run-uid"
const AttemptLabel = "training.graft.dev/attempt"

// AttemptName is deterministic, DNS-safe, and independent of a user's long name.
func AttemptName(run *training.TrainingRun) string {
	uid := string(run.UID)
	if len(uid) > 12 {
		uid = uid[:12]
	}
	return fmt.Sprintf("graft-%s-a%d", uid, run.Status.Attempt)
}

func owner(run *training.TrainingRun) []metav1.OwnerReference {
	return []metav1.OwnerReference{*metav1.NewControllerRef(run, training.GroupVersion.WithKind("TrainingRun"))}
}

func ConfigMap(run *training.TrainingRun) (*corev1.ConfigMap, error) {
	data, err := json.Marshal(run.Spec)
	if err != nil {
		return nil, err
	}
	return &corev1.ConfigMap{ObjectMeta: metav1.ObjectMeta{
		Name: AttemptName(run), Namespace: run.Namespace, OwnerReferences: owner(run),
	}, Data: map[string]string{"config.json": string(data)}}, nil
}

func MasterService(run *training.TrainingRun) *corev1.Service {
	return &corev1.Service{ObjectMeta: metav1.ObjectMeta{
		Name: AttemptName(run), Namespace: run.Namespace, OwnerReferences: owner(run),
	}, Spec: corev1.ServiceSpec{ClusterIP: corev1.ClusterIPNone, PublishNotReadyAddresses: true,
		Selector: map[string]string{AttemptLabel: AttemptName(run), "training.graft.dev/role": "master"},
		Ports:    []corev1.ServicePort{{Name: "rendezvous", Port: 29400}},
	}}
}

// VolcanoJob delegates gang admission and node placement to Volcano. GRAFT owns
// the logical attempt and recovery policy, so both lower retry loops are disabled.
func VolcanoJob(run *training.TrainingRun) *unstructured.Unstructured {
	name := AttemptName(run)
	spec := run.Spec
	makeTask := func(role string, replicas int32) map[string]interface{} {
		limits := map[string]interface{}{"cpu": "2", "memory": "3Gi"}
		requests := map[string]interface{}{"cpu": "250m", "memory": "768Mi"}
		if spec.GPUsPerWorker > 0 {
			limits["nvidia.com/gpu"] = int64(spec.GPUsPerWorker)
			requests["nvidia.com/gpu"] = int64(spec.GPUsPerWorker)
		}
		command := fmt.Sprintf("exec python -m graft_worker.rl --config /etc/graft/config.json --run-dir /artifacts/%s --attempt %d", run.UID, run.Status.Attempt)
		if spec.Workload == "ddp" {
			rank := "0"
			if role == "worker" {
				rank = "$((${HOSTNAME##*-}+1))"
			}
			command = fmt.Sprintf("exec python -m torch.distributed.run --nnodes=%d --nproc-per-node=1 --node-rank=%s --rdzv-backend=c10d --rdzv-id=%s --rdzv-endpoint=%s:29400 --max-restarts=0 --monitor-interval=1 -m graft_worker.train --config /etc/graft/config.json --run-dir /artifacts/%s --attempt %d", spec.Replicas, rank, name, name, run.UID, run.Status.Attempt)
		}
		podSpec := map[string]interface{}{
			"restartPolicy": "Never", "terminationGracePeriodSeconds": int64(3),
			"securityContext": map[string]interface{}{"runAsUser": int64(10001), "runAsGroup": int64(10001), "fsGroup": int64(10001)},
			"containers": []interface{}{map[string]interface{}{
				"name": "trainer", "image": spec.Image, "imagePullPolicy": "IfNotPresent",
				"command":         []interface{}{"/bin/bash", "-c", command},
				"resources":       map[string]interface{}{"limits": limits, "requests": requests},
				"securityContext": map[string]interface{}{"allowPrivilegeEscalation": false, "capabilities": map[string]interface{}{"drop": []interface{}{"ALL"}}},
				"env": []interface{}{
					map[string]interface{}{"name": "PYTHONUNBUFFERED", "value": "1"},
					map[string]interface{}{"name": "OMP_NUM_THREADS", "value": "1"},
					map[string]interface{}{"name": "CUBLAS_WORKSPACE_CONFIG", "value": ":4096:8"},
					map[string]interface{}{"name": "NCCL_DEBUG", "value": "WARN"},
					map[string]interface{}{"name": "GRAFT_ATTEMPT", "value": strconv.Itoa(run.Status.Attempt)},
				},
				"volumeMounts": []interface{}{
					map[string]interface{}{"name": "artifacts", "mountPath": "/artifacts"},
					map[string]interface{}{"name": "config", "mountPath": "/etc/graft", "readOnly": true},
					map[string]interface{}{"name": "shm", "mountPath": "/dev/shm"},
				},
			}},
			"volumes": []interface{}{
				map[string]interface{}{"name": "artifacts", "persistentVolumeClaim": map[string]interface{}{"claimName": "graft-artifacts"}},
				map[string]interface{}{"name": "config", "configMap": map[string]interface{}{"name": name}},
				map[string]interface{}{"name": "shm", "emptyDir": map[string]interface{}{"medium": "Memory", "sizeLimit": "512Mi"}},
			},
		}
		if spec.GPUsPerWorker > 0 {
			podSpec["runtimeClassName"] = "nvidia"
		}
		return map[string]interface{}{"name": role, "replicas": int64(replicas), "maxRetry": int64(0),
			"template": map[string]interface{}{
				"metadata": map[string]interface{}{"labels": map[string]interface{}{RunLabel: string(run.UID), AttemptLabel: name, "training.graft.dev/role": role}},
				"spec":     podSpec,
			},
		}
	}
	tasks := []interface{}{makeTask("master", 1)}
	if spec.Replicas > 1 {
		tasks = append(tasks, makeTask("worker", spec.Replicas-1))
	}
	u := &unstructured.Unstructured{Object: map[string]interface{}{
		"apiVersion": VolcanoGVK.GroupVersion().String(), "kind": "Job",
		"metadata": map[string]interface{}{"name": name, "namespace": run.Namespace,
			"labels": map[string]interface{}{RunLabel: string(run.UID), AttemptLabel: name}},
		"spec": map[string]interface{}{
			"schedulerName": "volcano", "queue": spec.Queue, "priorityClassName": spec.PriorityClassName,
			"minAvailable": int64(spec.Replicas), "maxRetry": int64(0), "tasks": tasks,
			"policies": []interface{}{map[string]interface{}{"event": "PodFailed", "action": "AbortJob"}},
		},
	}}
	u.SetOwnerReferences(owner(run))
	return u
}
