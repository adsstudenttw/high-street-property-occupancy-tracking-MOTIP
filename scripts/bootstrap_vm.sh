#!/usr/bin/env bash
set -euo pipefail

# One-time bootstrap for Ubuntu 22.04 CUDA VMs.
# Installs: base tools, Docker, optionally NVIDIA container toolkit.

MODE="${1:-auto}"   # auto | cpu | gpu
if [[ "${MODE}" != "auto" && "${MODE}" != "cpu" && "${MODE}" != "gpu" ]]; then
  echo "Usage: $0 [auto|cpu|gpu]"
  exit 1
fi

if [[ "${EUID}" -eq 0 ]]; then
  echo "Run this script as a normal user (it uses sudo when needed)."
  exit 1
fi

if ! grep -q "22.04" /etc/os-release; then
  echo "Warning: this script is intended for Ubuntu 22.04."
fi

echo "[1/6] Installing base packages..."
sudo apt-get update
sudo apt-get install -y \
  ca-certificates \
  curl \
  git \
  gnupg \
  lsb-release

ENABLE_NVIDIA_TOOLKIT="false"
if [[ "${MODE}" == "gpu" ]]; then
  ENABLE_NVIDIA_TOOLKIT="true"
elif [[ "${MODE}" == "cpu" ]]; then
  ENABLE_NVIDIA_TOOLKIT="false"
else
  if command -v nvidia-smi >/dev/null 2>&1; then
    ENABLE_NVIDIA_TOOLKIT="true"
  fi
fi

echo "[2/6] Checking NVIDIA driver..."
if [[ "${ENABLE_NVIDIA_TOOLKIT}" == "true" ]]; then
  if command -v nvidia-smi >/dev/null 2>&1; then
    nvidia-smi || true
  else
    echo "nvidia-smi not found but GPU mode requested. Install NVIDIA drivers first."
    exit 1
  fi
else
  echo "CPU mode: skipping NVIDIA driver/toolkit requirements."
fi

echo "[3/6] Installing Docker (official Docker apt repo)..."
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

sudo systemctl enable docker
sudo systemctl start docker
sudo usermod -aG docker "${USER}"

echo "[4/6] Installing NVIDIA Container Toolkit..."
if [[ "${ENABLE_NVIDIA_TOOLKIT}" == "true" ]]; then
  distribution=$(. /etc/os-release; echo "${ID}${VERSION_ID}")
  curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey \
    | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
  curl -s -L "https://nvidia.github.io/libnvidia-container/${distribution}/libnvidia-container.list" \
    | sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' \
    | sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list >/dev/null

  sudo apt-get update
  sudo apt-get install -y nvidia-container-toolkit
  sudo nvidia-ctk runtime configure --runtime=docker
  sudo systemctl restart docker
else
  echo "CPU mode: skipping NVIDIA container toolkit install."
fi

echo "[5/6] Verifying Docker access..."
docker --version
if [[ "${ENABLE_NVIDIA_TOOLKIT}" == "true" ]]; then
  docker run --rm --gpus all nvidia/cuda:12.1.1-base-ubuntu22.04 nvidia-smi
fi

echo "[6/6] Done."
echo "Open a new shell session (or run: newgrp docker) before using docker without sudo."
