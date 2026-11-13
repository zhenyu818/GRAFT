// Package v1alpha1 defines the declarative GRAFT training API.
package v1alpha1

import (
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/runtime/schema"
)

var GroupVersion = schema.GroupVersion{Group: "training.graft.dev", Version: "v1alpha1"}

// FaultSpec selects a reproducible, first-attempt process-level experiment.
type FaultSpec struct {
	Kind string `json:"kind,omitempty"`
	Step int    `json:"step,omitempty"`
	Rank int    `json:"rank,omitempty"`
}

type TrainingRunSpec struct {
	Workload                  string    `json:"workload"`
	Image                     string    `json:"image"`
	Replicas                  int32     `json:"replicas"`
	GPUsPerWorker             int32     `json:"gpusPerWorker"`
	PriorityClassName         string    `json:"priorityClassName"`
	Queue                     string    `json:"queue"`
	Steps                     int       `json:"steps"`
	Seed                      int       `json:"seed"`
	MicroBatch                int       `json:"microBatch"`
	Accumulation              int       `json:"accumulation"`
	CheckpointEvery           int       `json:"checkpointEvery"`
	ProgressTimeoutSeconds    int       `json:"progressTimeoutSeconds"`
	StartupTimeoutSeconds     int       `json:"startupTimeoutSeconds"`
	MaxRetries                int       `json:"maxRetries"`
	Suspend                   bool      `json:"suspend,omitempty"`
	StepDelaySeconds          float64   `json:"stepDelaySeconds,omitempty"`
	EnvironmentConcurrency    int       `json:"environmentConcurrency"`
	EnvironmentTimeoutSeconds float64   `json:"environmentTimeoutSeconds"`
	InjectEnvironmentTimeout  bool      `json:"injectEnvironmentTimeout,omitempty"`
	Fault                     FaultSpec `json:"fault,omitempty"`
}

type Recovery struct {
	Attempt          int         `json:"attempt"`
	Reason           string      `json:"reason"`
	DetectedAt       metav1.Time `json:"detectedAt"`
	DetectedAtUnixMs int64       `json:"detectedAtUnixMs"`
	LastStep         int         `json:"lastStep"`
}

type TrainingRunStatus struct {
	ObservedGeneration int64              `json:"observedGeneration,omitempty"`
	Phase              string             `json:"phase,omitempty"`
	Attempt            int                `json:"attempt"`
	JobName            string             `json:"jobName,omitempty"`
	SubmittedAt        *metav1.Time       `json:"submittedAt,omitempty"`
	StartedAt          *metav1.Time       `json:"startedAt,omitempty"`
	AttemptStartedAt   *metav1.Time       `json:"attemptStartedAt,omitempty"`
	FinishedAt         *metav1.Time       `json:"finishedAt,omitempty"`
	LastProgressAt     *metav1.Time       `json:"lastProgressAt,omitempty"`
	LastStep           int                `json:"lastStep"`
	NextRetryAt        *metav1.Time       `json:"nextRetryAt,omitempty"`
	WorkerUIDs         []string           `json:"workerUIDs,omitempty"`
	Recoveries         []Recovery         `json:"recoveries,omitempty"`
	Conditions         []metav1.Condition `json:"conditions,omitempty"`
}

type TrainingRun struct {
	metav1.TypeMeta   `json:",inline"`
	metav1.ObjectMeta `json:"metadata,omitempty"`
	Spec              TrainingRunSpec   `json:"spec"`
	Status            TrainingRunStatus `json:"status,omitempty"`
}

type TrainingRunList struct {
	metav1.TypeMeta `json:",inline"`
	metav1.ListMeta `json:"metadata,omitempty"`
	Items           []TrainingRun `json:"items"`
}

// DeepCopyInto isolates informer cache objects from reconciliation mutations.
func (in *TrainingRun) DeepCopyInto(out *TrainingRun) {
	*out = *in
	in.ObjectMeta.DeepCopyInto(&out.ObjectMeta)
	out.Status.SubmittedAt = in.Status.SubmittedAt.DeepCopy()
	out.Status.StartedAt = in.Status.StartedAt.DeepCopy()
	out.Status.AttemptStartedAt = in.Status.AttemptStartedAt.DeepCopy()
	out.Status.FinishedAt = in.Status.FinishedAt.DeepCopy()
	out.Status.LastProgressAt = in.Status.LastProgressAt.DeepCopy()
	out.Status.NextRetryAt = in.Status.NextRetryAt.DeepCopy()
	if in.Status.WorkerUIDs != nil {
		out.Status.WorkerUIDs = append([]string{}, in.Status.WorkerUIDs...)
	}
	if in.Status.Recoveries != nil {
		out.Status.Recoveries = append([]Recovery{}, in.Status.Recoveries...)
	}
	if in.Status.Conditions != nil {
		out.Status.Conditions = append([]metav1.Condition{}, in.Status.Conditions...)
	}
}
func (in *TrainingRun) DeepCopyObject() runtime.Object {
	if in == nil {
		return nil
	}
	out := new(TrainingRun)
	in.DeepCopyInto(out)
	return out
}
func (in *TrainingRunList) DeepCopyObject() runtime.Object {
	if in == nil {
		return nil
	}
	out := new(TrainingRunList)
	*out = *in
	in.ListMeta.DeepCopyInto(&out.ListMeta)
	if in.Items != nil {
		out.Items = make([]TrainingRun, len(in.Items))
		for i := range in.Items {
			in.Items[i].DeepCopyInto(&out.Items[i])
		}
	}
	return out
}
func AddToScheme(s *runtime.Scheme) error {
	s.AddKnownTypes(GroupVersion, &TrainingRun{}, &TrainingRunList{})
	metav1.AddToGroupVersion(s, GroupVersion)
	return nil
}
