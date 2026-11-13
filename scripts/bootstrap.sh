#!/usr/bin/env bash
# Install the pinned single-node Kubernetes platform on a dedicated GPU host.
set -euo pipefail
K3S_VERSION=v1.32.11+k3s1
VOLCANO_VERSION=v1.11.1
DEVICE_PLUGIN_VERSION=v0.17.0
TOOLKIT_VERSION=1.20.2-1
ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
TOOLS="$ROOT/.work/tools"
mkdir -p "$TOOLS"
command -v nvidia-smi >/dev/null
apt-get update
DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends ca-certificates curl gnupg docker.io make
curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | gpg --dearmor --yes -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
curl -fsSL https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list | sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' > /etc/apt/sources.list.d/nvidia-container-toolkit.list
apt-get update
DEBIAN_FRONTEND=noninteractive apt-get install -y \
  "nvidia-container-toolkit=$TOOLKIT_VERSION" "nvidia-container-toolkit-base=$TOOLKIT_VERSION" \
  "libnvidia-container-tools=$TOOLKIT_VERSION" "libnvidia-container1=$TOOLKIT_VERSION"
systemctl enable --now docker
curl -fL --retry 3 "https://github.com/k3s-io/k3s/releases/download/$K3S_VERSION/k3s" -o "$TOOLS/k3s"
curl -fL --retry 3 "https://github.com/k3s-io/k3s/releases/download/$K3S_VERSION/sha256sum-amd64.txt" -o "$TOOLS/k3s-checksums.txt"
(cd "$TOOLS" && awk '$2 == "k3s" {print}' k3s-checksums.txt | sha256sum -c -)
install -m 0755 "$TOOLS/k3s" /usr/local/bin/k3s
curl -fsSL "https://raw.githubusercontent.com/k3s-io/k3s/$K3S_VERSION/install.sh" -o "$TOOLS/install-k3s.sh"
INSTALL_K3S_SKIP_DOWNLOAD=true INSTALL_K3S_VERSION="$K3S_VERSION" sh "$TOOLS/install-k3s.sh" server \
  --disable traefik --disable servicelb --disable metrics-server --write-kubeconfig-mode 600
install -d -m 0770 -o 10001 -g 10001 /var/lib/graft/artifacts
kubectl wait --for=condition=Ready node --all --timeout=180s
kubectl apply -f "https://raw.githubusercontent.com/volcano-sh/volcano/$VOLCANO_VERSION/installer/volcano-development.yaml"
kubectl apply -f "https://raw.githubusercontent.com/NVIDIA/k8s-device-plugin/$DEVICE_PLUGIN_VERSION/deployments/static/nvidia-device-plugin.yml"
kubectl -n kube-system patch daemonset nvidia-device-plugin-daemonset --type merge \
  -p '{"spec":{"template":{"spec":{"runtimeClassName":"nvidia"}}}}'
kubectl -n kube-system rollout status daemonset/nvidia-device-plugin-daemonset --timeout=180s
kubectl -n volcano-system rollout status deployment/volcano-scheduler --timeout=300s
kubectl -n volcano-system rollout status deployment/volcano-controllers --timeout=300s
kubectl -n volcano-system rollout status deployment/volcano-admission --timeout=300s
kubectl get nodes -o custom-columns=NAME:.metadata.name,GPUS:.status.allocatable.nvidia\\.com/gpu
printf 'GRAFT platform installation completed.\n'
