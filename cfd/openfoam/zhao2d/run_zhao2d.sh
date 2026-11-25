#!/bin/bash
# run_zhao2d.sh — G2 re-validation at the Zhao 2008 conditions, star units (h1 = 1), Re = 12.2.
# Runs BirdCarreau then Newtonian, timing each run (wall seconds). Run on the VM inside ~/of/zhao2d.
set +eu; . /usr/lib/openfoam/openfoam2306/etc/bashrc || true; set -eu
cd "$(dirname "$0")"

blockMesh > log.blockMesh 2>&1
checkMesh > log.checkMesh 2>&1
grep -E "^\s*cells:" log.checkMesh | head -1 || true

run_case () {  # $1 = label
    rm -rf postProcessing [1-9]* 2>/dev/null || true
    TIMEFORMAT="wall_seconds %R"
    { time simpleFoam > "log.simpleFoam.$1" 2>&1; } 2> "log.time.$1"
    tail -1 "log.time.$1"
    grep -E "ExecutionTime" "log.simpleFoam.$1" | tail -1
    mv postProcessing "postProcessing.$1"
}

# Carreau (default constant/transportProperties)
run_case carreau
# Newtonian
cp constant/transportProperties constant/transportProperties.carreau
cp constant/transportProperties.newtonian constant/transportProperties
run_case newtonian
cp constant/transportProperties.carreau constant/transportProperties
echo RUNS_DONE
