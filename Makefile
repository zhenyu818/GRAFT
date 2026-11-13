PYTHON ?= python3
KUBECTL ?= kubectl
VERSION ?= 0.1.0

.PHONY: build test images load install experiment report

build:
	mkdir -p bin
	CGO_ENABLED=0 go build -trimpath -o bin/graft-manager ./cmd/manager

test:
	go vet ./...
	go test -race ./...
	PYTHONPATH=python $(PYTHON) -m unittest discover -s tests -v

images: build
	docker build -f build/controller.Dockerfile -t graft-controller:$(VERSION) .
	docker build -f build/worker.Dockerfile -t graft-worker:$(VERSION) .

load:
	docker save graft-controller:$(VERSION) graft-worker:$(VERSION) | k3s ctr images import -

install:
	$(KUBECTL) apply -f config/crd.yaml
	$(KUBECTL) apply -f config/platform.yaml
	$(KUBECTL) -n graft-system rollout status deployment/graft-controller --timeout=180s

experiment:
	PYTHONPATH=python $(PYTHON) scripts/experiments.py --output results/dual-v100

report:
	$(PYTHON) scripts/report.py results/dual-v100/results.json
