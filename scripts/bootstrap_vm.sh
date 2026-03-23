#!/usr/bin/env bash
set -euo pipefail

# One-time bootstrap for Ubuntu 22.04 CUDA VMs.
# Installs: base tools, Docker, and NVIDIA container toolkit.
# Configures Docker and containerd storage roots under a user-provided path,
# which is intended to live on a mounted SURF volume instead of the root disk.

if [[ "${1:-}" == "gpu" ]]; then
  shift
fi
if [[ "$#" -gt 1 ]]; then
  echo "Usage: $0 [gpu] [storage-root]"
  exit 1
fi
STORAGE_ROOT="${1:-$(pwd)/.surf-storage}"

if [[ "${EUID}" -eq 0 ]]; then
  echo "Run this script as a normal user (it uses sudo when needed)."
  exit 1
fi

if ! grep -q "22.04" /etc/os-release; then
  echo "Warning: this script is intended for Ubuntu 22.04."
fi

mkdir -p "${STORAGE_ROOT}"
STORAGE_ROOT="$(cd "${STORAGE_ROOT}" && pwd)"
DOCKER_ROOT="${STORAGE_ROOT}/docker"
CONTAINERD_ROOT="${STORAGE_ROOT}/containerd"
DOCKER_TMPDIR="${STORAGE_ROOT}/docker-tmp"
CACHE_ROOT="${STORAGE_ROOT}/cache"
TMP_ROOT="${STORAGE_ROOT}/tmp"
OPTUNA_ROOT="${STORAGE_ROOT}/optuna"
MLRUNS_ROOT="${STORAGE_ROOT}/mlruns"

mkdir -p \
  "${DOCKER_ROOT}" \
  "${CONTAINERD_ROOT}" \
  "${DOCKER_TMPDIR}" \
  "${CACHE_ROOT}" \
  "${TMP_ROOT}" \
  "${OPTUNA_ROOT}" \
  "${MLRUNS_ROOT}"

ROOT_DEVICE="$(df -P / | awk 'NR==2 {print $1}')"
STORAGE_DEVICE="$(df -P "${STORAGE_ROOT}" | awk 'NR==2 {print $1}')"

echo "Storage root: ${STORAGE_ROOT}"
if [[ "${ROOT_DEVICE}" == "${STORAGE_DEVICE}" ]]; then
  echo "Warning: STORAGE_ROOT is on the same filesystem as '/'."
  echo "Make sure you run this script from the mounted SURF volume if you want Docker data off the root disk."
fi

echo "[1/7] Installing base packages..."
sudo apt-get update
sudo apt-get install -y \
  ca-certificates \
  curl \
  git \
  gnupg \
  lsb-release \
  python3

echo "[2/7] Checking NVIDIA driver..."
if command -v nvidia-smi >/dev/null 2>&1; then
  nvidia-smi || true
else
  echo "nvidia-smi not found. Install the NVIDIA driver stack first."
  exit 1
fi

echo "[3/7] Installing Docker (official Docker apt repo)..."
sudo install -m 0755 -d /etc/apt/keyrings
curl -fsSL https://download.docker.com/linux/ubuntu/gpg \
  | sudo gpg --dearmor -o /etc/apt/keyrings/docker.gpg
sudo chmod a+r /etc/apt/keyrings/docker.gpg

source /etc/os-release
echo \
  "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.gpg] https://download.docker.com/linux/ubuntu ${VERSION_CODENAME} stable" \
  | sudo tee /etc/apt/sources.list.d/docker.list >/dev/null

sudo apt-get update
sudo apt-get install -y \
  docker-ce \
  docker-ce-cli \
  containerd.io \
  docker-buildx-plugin \
  docker-compose-plugin

echo "[4/7] Configuring Docker and containerd storage..."
sudo install -d -m 0755 /etc/containerd
if [[ -f /etc/containerd/config.toml ]]; then
  sudo cp /etc/containerd/config.toml /etc/containerd/config.toml.bak
fi
sudo containerd config default | sudo tee /etc/containerd/config.toml >/dev/null
sudo sed -i "s#^root = .*#root = '${CONTAINERD_ROOT}'#" /etc/containerd/config.toml

sudo install -d -m 0755 /etc/docker
if [[ -f /etc/docker/daemon.json ]]; then
  sudo cp /etc/docker/daemon.json /etc/docker/daemon.json.bak
fi
cat <<EOF | sudo tee /etc/docker/daemon.json >/dev/null
{
  "data-root": "${DOCKER_ROOT}",
  "features": {
    "buildkit": true
  }
}
EOF

sudo install -d -m 0755 /etc/systemd/system/docker.service.d
cat <<EOF | sudo tee /etc/systemd/system/docker.service.d/storage-root.conf >/dev/null
[Service]
Environment="DOCKER_TMPDIR=${DOCKER_TMPDIR}"
EOF
sudo systemctl daemon-reload

echo "[5/7] Installing NVIDIA Container Toolkit..."
distribution=$(. /etc/os-release; echo "${ID}${VERSION_ID}")
curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey \
  | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
curl -s -L "https://nvidia.github.io/libnvidia-container/${distribution}/libnvidia-container.list" \
  | sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' \
  | sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list >/dev/null

sudo apt-get update
sudo apt-get install -y nvidia-container-toolkit
sudo nvidia-ctk runtime configure --runtime=docker
sudo apt-get clean
sudo rm -rf /var/lib/apt/lists/*

echo "[6/7] Starting services and verifying Docker access..."
sudo systemctl enable containerd
sudo systemctl restart containerd
sudo systemctl enable docker
sudo systemctl restart docker
sudo usermod -aG docker "${USER}"

docker --version
sudo docker info --format 'DockerRootDir={{.DockerRootDir}}'
echo "Containerd root: $(sudo awk -F"'" '/^root = / {print $2; exit}' /etc/containerd/config.toml)"
sudo docker run --rm --gpus all nvidia/cuda:12.1.1-base-ubuntu22.04 nvidia-smi

echo "[7/7] Done."
echo "Open a new shell session (or run: newgrp docker) before using docker without sudo."
echo "Repo/runtime storage root: ${STORAGE_ROOT}"
