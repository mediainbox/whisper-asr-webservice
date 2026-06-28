# Build/release helpers for the Mediainbox GPU image.
#
# The GPU image is linux/amd64 (CUDA). Building it locally on an arm64 Mac uses QEMU
# emulation, which makes the torch+CUDA install painfully slow. Prefer Cloud Build, which
# builds natively on amd64 and pushes straight to Artifact Registry.

REGISTRY ?= us-central1-docker.pkg.dev/cloud-logger-177603/mediainbox/whisper-asr-webservice-gpu
# Version comes from pyproject.toml; override with `make build-gpu VERSION=x.y.z`.
VERSION  ?= $(shell sed -n 's/^version = "\(.*\)"/\1/p' pyproject.toml)

.PHONY: build-gpu build-gpu-local

## Build the GPU image on Google Cloud Build (native amd64) and push :$(VERSION) + :latest.
build-gpu:
	gcloud builds submit --config cloudbuild.gpu.yaml \
	  --substitutions=_REGISTRY=$(REGISTRY),_VERSION=$(VERSION) .

## Fallback: build locally with buildx (slow under QEMU on arm64; use only without gcloud).
build-gpu-local:
	docker buildx build --platform linux/amd64 -f Dockerfile.gpu \
	  -t $(REGISTRY):$(VERSION) -t $(REGISTRY):latest --push .
