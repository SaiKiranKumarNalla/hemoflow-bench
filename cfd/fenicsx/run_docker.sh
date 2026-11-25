#!/usr/bin/env bash
# Run a script inside the FEniCSx container with the repo mounted at /work.
# Usage: cfd/fenicsx/run_docker.sh cfd/fenicsx/bench_kg.py [args]
# DOCKER_CONFIG may point to an empty config (anonymous pulls, no keychain helper).
set -euo pipefail
IMAGE=${FENICSX_IMAGE:-ghcr.io/fenics/dolfinx/dolfinx:v0.9.0}
REPO=$(cd "$(dirname "$0")/../.." && pwd)
exec docker run --rm -e SUPG="${SUPG:-1}" -e FORM="${FORM:-comsol}" -e PHILLIPS="${PHILLIPS:-0}" -e WALL_LIFT="${WALL_LIFT:-0}" -e FARFIELD="${FARFIELD:-0}" -e ALPHAS="${ALPHAS:-0.01,0.1,0.2,0.3,0.4,0.5}" -v "$REPO":/work -w /work "$IMAGE" python3 "$@"
