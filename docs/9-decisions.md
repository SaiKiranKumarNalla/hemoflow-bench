# 9 · Design decisions log

Dated modelling and engineering decisions.

## Contents

- [Decisions log (newest first)](#decisions-log-newest-first)
- [Roadmap (target: apply 2026-11-20)](#roadmap-target-apply-2026-11-20)

---

## Decisions log (newest first)


- 2026-11-24 — M2b Euler–Euler split, Wu stress factor & steady-state criterion:
  - **M2b-A** = stock `multiphaseEulerFoam`, `WenYu` drag, constant effective RBC viscosity (Wu 2015 Yeleswarapu fallback), **no lift** (20 anchors in `campaign/m2b_design.jsonl`). Article will state: no CFL from lift in G1; expansion recirculation/depletion comparable.
  - **M2b-B** = stock `reactingTwoPhaseEulerFoam` (`rhoConst` isothermal, `noBlending` with continuous `plasma`, `WenYu` drag) with stock **`LegendreMagnaudet`** lift (verified directly in ESI OpenFOAM-v2306 source to reduce analytically to Saffman shear-lift scaling in the low-$\text{Re}_p$ microfluidic regime; optional `constantCoefficient` with Saffman-matched $C_L$ also supported), same 20 anchors (`campaign/m2b_lift_design.jsonl`).
  - Custom solver with exact Wu 2015 closures (Rusche–Issa drag, exact Saffman tensor lift, local Yeleswarapu viscosity): **deferred**.
  - Wu 2015 Eq. (6)/(13) stress factor: keep `stress_factor_half=True` (faithful to the published equations so Wu's results are reproducible; documented in [§2-theory](2-theory.md)).
  - M2b transient convergence criterion: run to steady state = relative change of $\Delta p^*$, outlet $\phi$-profile mean, and reattachment length (for expansion families `G2`, `KG`) `< 1e-4` over the last 10 % of simulated time (`converged_by: "steady_10pct"`); otherwise `status: "failed", reason: "not_steady"`.
- 2026-11-24 — Convergence criterion: a `simpleFoam` run (`residualControl` `1e-5`
  for `p` and `U`, `endTime 5000`, `fvSchemes` plain `Gauss linear` without `cellLimited` limiter, `nNonOrthogonalCorrectors 0`
  unless bifurcation/serpentine `checkMesh` non-orthogonality `> 40°`) is `converged` if
  **(a)** all initial residuals `< 1e-5` (`converged_by: "residual"`), **or**
  **(b)** all initial residuals `< 1e-3`, continuity error (`sum local`) `< 1e-8`, and both $\Delta p^*$
  (`inletPressure` patch `areaAverage(p)`) and outlet-centreline velocity (`qoiProbes` `|U|`) change by `< 1e-5`
  relative over the last 500 iterations (`converged_by: "qoi"`). Runs meeting neither are recorded as
  `status: "failed", reason: "not_converged"` and never silently kept.
- 2026-11-24 — Physics: three model families on the same geometries — M1 single-phase non-Newtonian (Carreau),
  M2 continuum multiphase (mixture + Phillips migration; Euler–Euler optional), M3 particulate RBCs (DPMFoam
  rigid spheres; HemoCell later) as comparison. VOF/interFoam dropped: blood has no sharp interface (see
  [§2-theory](2-theory.md) §3.3). Equations in [§2-theory](2-theory.md).
- 2026-11-24 — Scope to 2026-11-28: straight channel + sudden expansion, 2D; M1 validated CFD → small dataset
  (20–40 runs) → one surrogate → explain end to end. M2/M3, 2.5D/3D, FNO/DeepONet benchmark, MC, inverse design after.
- 2026-11-24 — Compute: GCP SPOT allowed (e2-medium, europe-west4, max-run 4 h, auto-DELETE, delete same session,
  log cost). OrbStack later (Mac has ~19 GB free disk). Overrides the old "no paid VMs" rule.
- 2026-11-24 — Tools: whole project from the team IDE; the team codes, the team + the team tutor/review.
- 2026-11-24 — **Consistent dimensional mapping:** star runs stay nondimensional, but every run carries
  its own physical scale. Carreau: h = √(Re μ∞ λ/(ρ Cu)), U = √(Re μ∞ Cu/(ρ λ)), τ_ref = μ∞ Cu/λ, t_ref = λ/Cu.
  Newtonian: h is a free choice (default 50 µm). Window and hemolysis use the run's own scale, never a fixed 2023 τ*.
  The original M1 design (Cu ∈ [1, 1e5] independent of Re) puts 209/348 Carreau runs in mm-scale channels → added a
  device-space design `m1b` (h, U, blood params sampled, groups derived) in task T7.
- 2026-11-24 — **Encapsulation index normalization (provisional, the team; owner may override):** Eq. 1 maxima are taken
  on the developed inlet channel (−3 < x* < −1), not over the domain — the domain max of τ is mesh-dependent (corner
  singularity, GCI study). Then I > 1 means "better than the inlet channel". With φ uniform (M1) the inlet peak is
  exactly 2/(3√3) ≈ 0.385 (plane or tube Poiseuille), matching the 2023 range 0.2 < I < 0.4. I is post-processed from
  stored fields (no re-runs). Corpus targets add: flux-weighted RBC fraction with 0.2 ≤ I and τ in the window.
- 2026-11-24 — **Full-case archiving (owner):** every run stores mesh (`polyMesh`), case dicts, final fields, logs and
  postProcessing as `corpus/v1/cases/<tier>/<family>/<run_id>.tar.gz` next to the npz, plus a `code_version` hash in
  the manifest. Runs finished before this (validation, GCI, first M1 batch) are re-run with archiving (task T7).
- 2026-11-24 — **Stage 1 = full 2D CFD corpus** per [3-methods](3-methods.md) (≈600 runs, 7 geometry families, M1/M2/M2b/M3,
  GCI study). Supersedes the 20–40-run scope above. All runs and results in **nondimensional star units**.
  Write-up + scientific article after the corpus. AI after the sims; 3D only if 2D works.
- 2026-11-24 — Orchestration: the team (agy) and the team (vibe) run headless with **auto-approve, budget-capped**;
  the team orchestrates and reviews at each gate. Agents do not commit; the owner commits.
- 2026-11-24 — Compute budget Stage 1: **hard cap $10 GCP total**. Pilot on e2-medium SPOT, measure s/run, then choose
  e2-medium or an 8-vCPU SPOT VM for batches (all other SPOT rules unchanged). Max one VM per agent at a time.
- 2026-11-24 — Storage: new private bucket `gs://<corpus-bucket>` (europe-west4, standard, uniform access,
  public access prevented). The existing `gs://<other-bucket>` is another project — never touch it.
- 2026-11-24 — Pilot (T0): 30.4k cells, 2500 it: 151 s (Carreau) / 232 s (Newtonian) on e2-medium. Batch VMs:
  **n2-highcpu-8 SPOT** (europe-west4, same SPOT rules), max one per agent, 1 case per core. Stage-1 cap stays $10.
- 2026-11-24 — Validation case = Karino & Goldsmith 1977 tubular expansion via Wu 2015 §3.1 (axisymmetric wedge),
  not Zhao 2008 (see [§4-validation](4-validation.md) flag 1). Zhao 2008 stays a qualitative CFL comparison for M2b.

- 2026-11-08 — Scope is Stage A (device encapsulation) only; Stage B deferred.
- 2026-11-08 — Multi-fidelity: L1 2D finite-volume mixture model (CASSOULET 2D) for bulk data; L2 3D lattice
  Boltzmann in JAX on Colab for high fidelity; L3 HemoCell for calibration later.
- 2026-11-08 — Release: only 2D/2.5D CASSOULET code; no 3D, no exact 3D BSREM, no real scanner parameters.
- 2026-11-08 — Compute: no new paid services; existing ~175 Colab units; A100 for tested production batches only.

- 2026-11-24 — Corpus v2 (owner): paired 4-variant design — every design (geometry, h, U, haematocrit 0.25/0.45/0.55)
  runs as V1 Newtonian, V2 Carreau, V3 Euler–Euler drag+lift, V4 Euler–Euler full, same mesh, common plasma-viscosity
  scale. Lagrangian (DPMFoam) dropped, M1b and Phillips mixture replaced, LBM-IBM parked. See CORPUS_PLAN §6.

- 2026-11-24 — Corpus v2.2 (owner): Euler–Euler V3/V4 production in an own steady FEniCSx two-fluid solver;
  OpenFOAM Euler–Euler only for gates, the M2b-A control and a random 10 % cross-code sample at the end. CORPUS_PLAN §6.1–6.2.

- 2026-11-24 — Haematocrit φ ∈ [0.01, 0.6] with device-regime designs 0.01–0.1 and per-run carrier viscosity (owner);
  shear window = screening (W1 30–60, W2 15–45, W3 1–7 Pa) + time-aware dose metrics D15/D30, Giersiepen HI (mean, P95)
  and derived τ_hi,eff; RBC-flux weighted (owner + the team + the team + the team). [§4-validation](4-validation.md), CORPUS_PLAN §6.3.

- 2026-11-24 — Stage-1 compute cap raised from $10 to $15 (owner). Spent so far ≈ $0.35.

- 2026-11-24 — Two VMs in parallel (owner): the team T7 (OpenFOAM, rbc-team-4) + the team T8 (FEniCSx pipeline, rbc-team-1); one VM per agent; the team keeps the FEniCSx solver core + validation.

- 2026-11-24 — AWS dropped for now (owner): CLI installed (~/.local/share/aws-cli, profile default, eu-north-1) but the GitHub-account login did not complete; $100 AWS credit available later. Compute stays GCP (32-CPU trial quota) + Mac.

- 2026-11-24 — V3 = 2023 COMSOL setup (solid particles, Schiller–Naumann, Krieger) with Wu properties and an RBC
  packing limit φ_max = 0.966 (= 2/k∞, Quemada) instead of COMSOL's 0.62 (owner). V4 = V3 structure + Wu/Sezer
  closures, solved with COMSOL's documented pseudo-time stepping + damped Newton ([§2-theory](2-theory.md) §5).

- 2026-11-24 — Paired design: sample nominal wall shear τ_w log-uniform 2–150 Pa (derive U), reject Re > 60, keep 164
  designs. V1/V2 production launched in parallel with V4
  validation (owner) — the team T10.

- 2026-11-24 — Publishing (owner): new PUBLIC repo `hemoflow-bench` from a cleaned snapshot of this repo (all except
  `course/`; no emails / GCP ids / bucket names; fresh history) after the team review + release check. `rbc-shear-surrogate`
  stays PRIVATE and untouched (working repo). `cassoulet-showcase` made private. 2023 COMSOL figures/results may be
  published (owner), credited to co-author Gereige; collaborator eliogereige invited (write) to rbc-shear-surrogate,
  to be added to hemoflow-bench at creation.

### CASSOULET allowlist (to be filled with the owner)
- `TODO`

---

## Roadmap (target: apply 2026-11-20)


Day-by-day detail: [DAILY_PLAN.md](DAILY_PLAN.md) (Day 0 = 2026-11-09).

| Dates | Milestone | Skills |
|---|---|---|
| until start (owner) | Find 2023 COMSOL files; supervisor OK for 2D CASSOULET release; read PINN/FNO/DeepONet | literature-update |
| Day 1 | Environments ready (Mac, VM + SLURM, Colab) | setup-environment |
| Days 2–3 | L1 mixture solver validated (224 µm, 0.56R, Wu 2015, CFL) | validate-solver |
| Day 4 | L1 campaign (~2,100 runs) + dataset card | run-sim-campaign |
| Days 5–6 | S0 + S1 (MLP, U-Net, FNO) + evaluation | train-surrogate |
| Day 7 | MC-1 per-cell loading; MC-2 uncertainty | monte-carlo |
| Day 8 | Inverse design for 3 synthetic patients | inverse-design |
| Days 8–10 (parallel, GPU) | L2 3D LBM validated, ~100–300 runs, scaling plot, S2 fine-tune | lbm-3d-gpu, train-surrogate |
| Day 11 | README, figures, limitations; release guard; publish | release-guard |
| Day 12 (2026-11-20) | CV, cover letter, apply | — |

Must-have: L1, S0, S1, MC-2, inverse design. Should-have: L2 + S2. After deadline: L3, MC-3, Stage B.
