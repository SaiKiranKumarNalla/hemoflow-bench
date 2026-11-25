"""Colab CPU smoke test for the FEniCSx two-fluid solver (run with `colab exec -f`).

Expects twofluid.py, check_channel.py, bench_kg.py uploaded to /work/cfd/fenicsx/ (the scripts' import path).
Installs FEniCSx via FEM-on-Colab if missing, then runs the straight-channel gate and the Karino-Goldsmith benchmark.
"""
import os
import subprocess
import sys
import time

t0 = time.time()
try:
    import dolfinx  # noqa: F401
except ImportError:
    subprocess.run("wget -q https://fem-on-colab.github.io/releases/fenicsx-install-release-real.sh "
                   "-O /tmp/fenicsx-install.sh && bash /tmp/fenicsx-install.sh", shell=True, check=True)
import dolfinx  # noqa: E402

print(f"dolfinx {dolfinx.__version__}  install+import {time.time() - t0:.0f}s  cpus={os.cpu_count()}", flush=True)
print(subprocess.run("free -g | head -2; lscpu | grep 'Model name'", shell=True, capture_output=True, text=True).stdout)
env = dict(os.environ, SUPG="0")
for script, args in (("check_channel.py", []), ("bench_kg.py", ["20", "plasma"])):
    t1 = time.time()
    out = subprocess.run([sys.executable, f"/work/cfd/fenicsx/{script}", *args], env=env, capture_output=True, text=True)
    lines = [ln for ln in (out.stdout + out.stderr).splitlines()
             if any(k in ln for k in ("phi_in", "Re_p=", "RESULT", "mesh", "Error"))]
    print(f"== {script} rc={out.returncode} {time.time() - t1:.0f}s", *lines, sep="\n", flush=True)
