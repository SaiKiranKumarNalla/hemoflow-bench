#!/usr/bin/env bash
# Run a generated nondimensional OpenFOAM v2306 case:
#   blockMesh -> checkMesh (fail on errors; non-ortho check) -> simpleFoam -> crash guard -> postProcess
# Exits non-zero on mesh errors (2), non-zero solver exit (3), or fatal/FPE/NaN crash (4).
# Convergence (residual < 1e-5 OR QoI steady over last 500 iters) is evaluated by campaign/run_batch.py.
#
# Usage:
#   bash run_case.sh                              # run inside the current case directory
#   bash run_case.sh /path/case                   # run the specified case directory
#   bash run_case.sh --check-log /path/log        # run only the crash guard on a log file
#   bash run_case.sh --configure-non-ortho /case  # run only the non-orthogonality fvSolution check

set -euo pipefail

check_simplefoam_log() {
    local log_file="$1"
    if [ ! -f "$log_file" ]; then
        echo "ERROR: simpleFoam log not found: $log_file" >&2
        return 4
    fi

    # Detect real crashes only; ignore OpenFOAM's startup banner:
    #   trapFpe: Floating point exception trapping enabled (FOAM_SIGFPE).
    if ! awk '
        BEGIN { found = 0; printed = 0 }
        /^[[:space:]]*trapFpe:[[:space:]]*Floating point exception trapping enabled/ { next }
        {
            hit = 0
            if ($0 ~ /FOAM FATAL/ ||
                $0 ~ /solution singularity/ ||
                $0 ~ /Floating point exception \(core dumped\)/ ||
                $0 ~ /(^|[^A-Za-z0-9_])sigFpe([^A-Za-z0-9_]|$)/) {
                hit = 1
            } else if ($0 ~ /Solving for |Initial residual|Final residual|continuity errors/) {
                low = tolower($0)
                if (low ~ /(^|[^a-z0-9_])[+-]?(nan|inf)([^a-z0-9_]|$)/) {
                    hit = 1
                }
            }
            if (hit) {
                found = 1
                if (printed < 20) {
                    print $0 > "/dev/stderr"
                    printed++
                }
            }
        }
        END { exit (found ? 4 : 0) }
    ' "$log_file"; then
        echo "ERROR: simpleFoam log shows fatal error, FPE crash, or NaN/Inf in residuals" >&2
        return 4
    fi
    return 0
}

configure_non_ortho_correctors() {
    local case_dir="$1"
    local log_cm="$case_dir/log.checkMesh"
    local params_json="$case_dir/params.json"
    local fv_sol="$case_dir/system/fvSolution"

    if [ ! -f "$log_cm" ] || [ ! -f "$fv_sol" ]; then
        return 0
    fi

    local family=""
    if [ -f "$params_json" ]; then
        family="$(awk -F'"' '/"family"[[:space:]]*:/ { print $4; exit }' "$params_json")"
    fi

    # Keep nNonOrthogonalCorrectors 0 for orthogonal families; only enable for
    # bifurcation / serpentine families (G4-G7) if checkMesh max non-orthogonality > 40 deg.
    case "$family" in
        G4|G5|G6|G7) ;;
        *) return 0 ;;
    esac

    local max_no
    max_no="$(awk '/Mesh non-orthogonality Max:/ { for (i=1; i<=NF; i++) if ($i == "Max:") { print $(i+1); exit } }' "$log_cm")"
    if [ -n "$max_no" ] && awk -v v="$max_no" 'BEGIN { exit !(v + 0 > 40.0) }'; then
        echo "INFO: family=$family checkMesh max non-orthogonality=${max_no} > 40 deg -> setting nNonOrthogonalCorrectors 1"
        awk '{ gsub(/nNonOrthogonalCorrectors[[:space:]]+[0-9]+;/, "nNonOrthogonalCorrectors 1;"); print }' "$fv_sol" > "$fv_sol.tmp"
        mv "$fv_sol.tmp" "$fv_sol"
    fi
}

if [ "${1:-}" = "--check-log" ]; then
    if [ $# -lt 2 ]; then
        echo "Usage: bash run_case.sh --check-log <log.simpleFoam>" >&2
        exit 2
    fi
    check_simplefoam_log "$2"
    exit $?
fi

if [ "${1:-}" = "--configure-non-ortho" ]; then
    configure_non_ortho_correctors "${2:-.}"
    exit 0
fi

CASE_DIR="${1:-.}"
cd "$CASE_DIR"

# OpenFOAM v2306 etc/bashrc references unset variables and may return 1 on a trailing check.
set +eu
if [ -f /usr/lib/openfoam/openfoam2306/etc/bashrc ]; then
    # shellcheck disable=SC1091
    . /usr/lib/openfoam/openfoam2306/etc/bashrc || true
fi
set -eu

echo "=== [1/4] blockMesh ($CASE_DIR) ==="
blockMesh > log.blockMesh 2>&1

echo "=== [2/4] checkMesh ==="
checkMesh -allGeometry -allTopology > log.checkMesh 2>&1
if grep -Eq "Failed [0-9]+ mesh checks|FOAM FATAL" log.checkMesh; then
    echo "ERROR: checkMesh reported mesh check failures:" >&2
    grep -E "Failed|ERROR|FOAM FATAL|non-orthogonality|skewness" log.checkMesh >&2 || true
    exit 2
fi
if ! grep -q "Mesh OK\." log.checkMesh; then
    echo "ERROR: checkMesh did not report 'Mesh OK.'" >&2
    tail -n 30 log.checkMesh >&2 || true
    exit 2
fi
configure_non_ortho_correctors "."

echo "=== [3/4] simpleFoam ==="
if ! simpleFoam > log.simpleFoam 2>&1; then
    echo "ERROR: simpleFoam exited non-zero" >&2
    tail -n 40 log.simpleFoam >&2 || true
    exit 3
fi

check_simplefoam_log log.simpleFoam

echo "=== [4/4] postProcess (writeCellCentres + writeCellVolumes + grad(U) + sample) ==="
postProcess -func writeCellCentres -latestTime > log.postProcess_C 2>&1
postProcess -func writeCellVolumes -latestTime > log.postProcess_V 2>&1
postProcess -func 'grad(U)' -latestTime > log.postProcess_gradU 2>&1
postProcess -func sample -latestTime > log.postProcess_sample 2>&1

if grep -q "SIMPLE solution converged" log.simpleFoam; then
    echo "=== DONE: $(grep 'SIMPLE solution converged' log.simpleFoam | tail -n 1) ==="
else
    echo "=== DONE: reached endTime (pending QoI convergence evaluation in run_batch) ==="
fi
