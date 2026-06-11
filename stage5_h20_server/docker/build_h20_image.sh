#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
IMAGE_TAG="${IMAGE_TAG:-detr-q3-h20:cu124}"

docker build -t "$IMAGE_TAG" "$SCRIPT_DIR"
docker image inspect "$IMAGE_TAG" --format '[ok] built {{.RepoTags}} size={{.Size}}'
