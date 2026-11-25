#!/bin/bash
# Tutor: run the 2D channel case on the GCP VM and validate vs analytic Poiseuille.
# Usage on VM:  bash run_channel2d.sh
#   1) source OpenFOAM  2) blockMesh (build grid)  3) simpleFoam (solve steady NS)
#   4) validate centerline profile against u(y) = 6 Uavg y (H-y) / H^2  (H=1, Uavg=2/3 -> Umax=1)
set -euo pipefail
set +eu; . /usr/lib/openfoam/openfoam2306/etc/bashrc || true; set -eu
cd "$(dirname "$0")/channel2d"
blockMesh > log.blockMesh 2>&1
nohup simpleFoam > log.simpleFoam 2>&1 &
SOLVER=$!
echo "simpleFoam pid $SOLVER — polling..."
while kill -0 $SOLVER 2>/dev/null; do
  tail -1 log.simpleFoam | sed 's/^/  /'
  sleep 20
done
wait $SOLVER || true
echo "=== solver finished; validating ==="
python3 ../validate_channel2d.py | tee ../validation_channel2d.txt
