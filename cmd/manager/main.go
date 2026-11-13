package main

import (
	"flag"
	"os"

	training "github.com/zhenyu818/GRAFT/api/v1alpha1"
	"github.com/zhenyu818/GRAFT/internal/controller"
	corev1 "k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/runtime"
	ctrl "sigs.k8s.io/controller-runtime"
	"sigs.k8s.io/controller-runtime/pkg/cache"
	"sigs.k8s.io/controller-runtime/pkg/healthz"
	"sigs.k8s.io/controller-runtime/pkg/log/zap"
	metricsserver "sigs.k8s.io/controller-runtime/pkg/metrics/server"
)

func main() {
	var namespace, artifacts, metricsAddress string
	var leader bool
	var concurrency int
	flag.StringVar(&namespace, "namespace", "graft-system", "Namespace managed by this controller")
	flag.StringVar(&artifacts, "artifacts", "/artifacts", "Shared artifact volume mount")
	flag.StringVar(&metricsAddress, "metrics-bind-address", ":8080", "Prometheus listener")
	flag.BoolVar(&leader, "leader-elect", true, "Elect one active manager through a Kubernetes Lease")
	flag.IntVar(&concurrency, "concurrency", 4, "Maximum concurrent reconciliations")
	logging := zap.Options{}
	logging.BindFlags(flag.CommandLine)
	flag.Parse()
	ctrl.SetLogger(zap.New(zap.UseFlagOptions(&logging)))
	scheme := runtime.NewScheme()
	must(corev1.AddToScheme(scheme))
	must(training.AddToScheme(scheme))
	mgr, err := ctrl.NewManager(ctrl.GetConfigOrDie(), ctrl.Options{
		Scheme: scheme, LeaderElection: leader, LeaderElectionID: "graft-controller.training.graft.dev", LeaderElectionNamespace: namespace,
		HealthProbeBindAddress: ":8081", Metrics: metricsserver.Options{BindAddress: metricsAddress},
		Cache: cache.Options{DefaultNamespaces: map[string]cache.Config{namespace: {}}},
	})
	must(err)
	r := &controller.Reconciler{Client: mgr.GetClient(), ArtifactRoot: artifacts, Concurrency: concurrency}
	must(r.SetupWithManager(mgr))
	must(mgr.AddHealthzCheck("healthz", healthz.Ping))
	must(mgr.AddReadyzCheck("readyz", healthz.Ping))
	must(mgr.Start(ctrl.SetupSignalHandler()))
}
func must(err error) {
	if err != nil {
		ctrl.Log.Error(err, "Manager stopped")
		os.Exit(1)
	}
}
