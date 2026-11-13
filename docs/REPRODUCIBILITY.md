# Reproducibility and Release Integrity

## Execution identity

`results/dual-v100/results.json` records the GPU model, memory, driver, Python,
NumPy, PyTorch, CUDA, NCCL, Kubernetes, container runtime, worker image identity,
and execution-source SHA-256. Configurations and event streams are stored alongside
each experiment's assertions and measurements.

Compute the execution fingerprint from a clean checkout:

```bash
python3 -c 'from scripts.experiments import source_digest; print(source_digest())'
```

The fingerprint covers API definitions, controller code, workers, tests, manifests,
container definitions, dependency files, and experiment/report scripts.

## File integrity

The publication allowlist selects source, configuration, documentation, and recorded
results. `release-manifest.json` stores each selected path, byte count, and SHA-256.
The manifest describes its payload; its own bytes are recorded by the Git commit.

Verify the file hashes:

```bash
python3 - <<'PY'
import hashlib
import json
from pathlib import Path
manifest = json.loads(Path('release-manifest.json').read_text())
for entry in manifest['entries']:
    data = Path(entry['path']).read_bytes()
    assert len(data) == entry['bytes'], entry['path']
    assert hashlib.sha256(data).hexdigest() == entry['sha256'], entry['path']
print('All published file hashes match.')
PY
```

`scripts/verify_artifact.py` also checks English-language content and sensitive-data
patterns. Positive controls exercise six text formats and filename inspection.

## Package verification

The release verification stages the allowlisted tree into an archive, extracts it
into a separate directory, and runs Go formatting, vet, and race-enabled tests.
The worker image installs the archived Python package into a fresh target directory
with networking disabled, checks the import location and dependency versions, and
runs the checkpoint, environment, and metrics protocol tests.

The accompanying `validation.json` records the checks and the tested source
fingerprint. The Kubernetes integration suite is reproduced with:

```bash
make experiment
make report
```
