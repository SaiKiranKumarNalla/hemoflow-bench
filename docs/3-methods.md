# 3 · Methods: model hierarchy, solvers, corpus design

How the simulations are built, run and organised.

## Contents

- [Model hierarchy (multi-fidelity)](#model-hierarchy-multi-fidelity)
- [Surrogate inputs and outputs](#surrogate-inputs-and-outputs)
- [Compute (no new paid services)](#compute-no-new-paid-services)
- [Stage 1 — 2D CFD corpus plan (v1, 2026-11-24)](#stage-1--2d-cfd-corpus-plan-v1-2026-11-24)

---

## Model hierarchy (multi-fidelity)


### Simulation levels
| Level | Model | Cost | Role | Hardware |
|---|---|---|---|---|
| L0 | Analytic: Poiseuille + Fåhræus + CFL estimate | µs | Sanity checks, surrogate features, initial guesses | Mac |
| **L1** | **2D mixture model** (Quemada/Carreau + Phillips) on CASSOULET 2D finite volume | s–min | Bulk training data (~2,000+ runs) | Ubuntu VM, parallel CPU |
| **L2** | **3D lattice Boltzmann** (D3Q19, MRT/regularised) + Hct transport, in JAX | ~30–80 s/run on A100 | High-fidelity set (~100–300 runs), verification | Colab A100/L4 |
| L3 | Cell-resolved LBM + immersed boundary (HemoCell) | hours | Calibrate pore-opening / damage thresholds (10–20 runs) | GPU, after deadline |

Cross-method check: L1 (finite volume) vs L2 run in 2D mode must agree on validation cases.

### Monte Carlo
| ID | What | Output |
|---|---|---|
| MC-1 | Lagrangian RBC tracking with random-walk shear-induced diffusion; per-cell shear dose and damage | Distribution of loading per cell (like flow cytometry) |
| MC-2 | Sample patient vectors around measured values → surrogate | Distribution of I and hemolysis per design → robust design |
| MC-3 | Bayesian calibration (MCMC) of thresholds and migration constants against published experiments | Parameters with uncertainty |

### Surrogates
| ID | Model | Trained on | Predicts |
|---|---|---|---|
| S0 | Gaussian process / MLP ensemble | L1 scalars | mean I, % cells in window, hemolysis, Δp (+ uncertainty) |
| S1 | FNO and U-Net (MLP baseline) | L1 fields | u, v, τ, φ, I fields |
| S2 | S1 pretrained on L1, fine-tuned (or Δ-learning) on L2 | L1 + L2 | 3D-corrected fields and scalars |

Inverse design: optimise geometry + flow on S0/S2 for a patient vector, constraint hemolysis < threshold, MC-2 for
robustness, verify the top 3 designs with L2. Active learning: add L1/L2 runs where ensemble members disagree.

---

## Surrogate inputs and outputs


### Inputs
| Group | Variables | Range | Encoding |
|---|---|---|---|
| Geometry | type ∈ {straight, sudden expansion, contraction, curved, bifurcation, pillar array}; D, L (or L/D), expansion ratio, bend radius, bifurcation angle | D 23–150 µm; L/D ≤ 20; angles 30–60° | Mask + signed distance field (image) + scalar channels |
| Flow | mean inlet velocity or flow rate; steady/pulsatile | V ≤ ~0.33 m/s; Re 0.01–12 | Scalar channels |
| Blood | patient vector from `04-blood-models.md` | as listed | Scalar channels |

### Outputs
- Fields (S1/S2): u, v (w in 3D), shear stress τ, haematocrit φ, encapsulation index I, accumulated damage.
- Scalars (S0 and heads): mean I in optimal zone, % cells with τ in window, hemolysis index, pressure drop, throughput.

### Data schema (HDF5, one file per run)
`/inputs/{geometry_params, mask, sdf, flow, blood}`, `/fields/{u,v,tau,phi,I,damage}` (float16), `/scalars/*`,
`/meta/{solver, level, version, seed, wall_time, config_hash}`. Every run is listed in `campaign/manifest.json`.

### Splits
train ~1,500 · val 200 · test 200 · **unseen** 200 (outside ranges + one held-out geometry type) · L2 ~100–300.

---

## Compute (no new paid services)


| Machine | Use |
|---|---|
| Mac (M1, 8 GB, ~23 GB free) | Code, git, small tests, plots. No CUDA (PyTorch `mps` only) |
| **GitHub Codespaces** (free: 120 core-hours/month ≈ 60 h on a 2-core, 8 GB RAM Linux machine; 15 GB storage) | Main Linux workspace: shell practice, SLURM (single node) job arrays, small L1 campaigns |
| **Kaggle notebooks, CPU** (free: 4 cores, ~30 GB RAM, sessions up to 12 h, runs continue in the background via "Save version") | Large L1 campaigns without using the Mac |
| **Google Cloud Shell** (free, small Linux VM in the browser, ~5 GB persistent home) | Quick Linux practice anywhere |
| University HPC (e.g. CALMIP, Toulouse) — only if the owner still has an account | Real SLURM cluster experience |
| Multipass Ubuntu VM on the Mac | Not used: it takes the Mac's own memory |
| Colab free / Kaggle (~30 GPU-h/week) | Debugging, training |
| Colab compute units (~175 left) | A100 ≈ 11–13 units/h (~14 h), L4 ≈ 4.5–5 (~36 h), T4 ≈ 2. Check exact rates in Colab |
| Kaggle Datasets / Hugging Face Hub | Moving datasets between VM and GPUs |

Budget: dev on free tier → L4 validation (10–15 units) → A100 production 200 L2 runs (30–60) → scaling plot (5–10)
→ training (20–30) → keep 50+ in reserve. Log units used in (internal planning notes).
Rules: disconnect idle runtimes; batch cases with `jax.vmap`; checkpoint to Drive/HF; float32 compute, float16 storage.

---

## Stage 1 — 2D CFD corpus plan (v1, 2026-11-24)


Merged from independent read-only proposals by the team and the team; the team orchestrates and reviews.
Owner directives: **all math nondimensional**; corpus in a GCS bucket; grid-independence studies; after the corpus,
a write-up **and a scientific article**; AI only after the sims; 3D only if 2D works.

### 0. Where the two agents agreed / differed
| Topic | the team | the team | Decision |
|---|---|---|---|
| Total runs | 500–720 | 600–700 | **~600** (tiers below) |
| M1 bulk | 400–600 | 400–500 | **400** (+ GCI) |
| M2 (mixture + Phillips) | 60–100 | 120–150 | **80**, after M1, as Δ(M2−M1) |
| M2b Euler–Euler | 15–25 anchors | 30–50 optional | **20 anchors**, closures must be sourced first |
| M3a DPMFoam | 0–10 | 20–30 | **≤ 10, φ ≤ 0.1, qualitative** (under-resolved at a/H = 0.056) |
| M3b deformable RBC | 8–15, own 2D JAX LBM-IBM on Colab | 5–10 calibration, JAX LBM-IBM | **8–10 calibration cases**, own 2D JAX LBM-IBM, Colab GPU |
| Sampling | scrambled Sobol | LHS | **Sobol** (extendable in powers of 2) |
| Meshing | blockMesh + Gmsh for bifurcations | blockMesh for all 8 families | **Python blockMesh generator**; Gmsh only if a family fails |
| Grid study | 3 levels, r = 1.5, GCI < 2 % | 3 levels, r = 2, GCI < 3 % | **3 levels, r = 1.5, Roache GCI (F_s = 1.25): < 2 % on Δp*, x_r/h, < 5 % on window/HI** |
| VM | n2/c2d-highcpu-8 SPOT | e2-medium SPOT | **pilot on e2-medium, measure s/run, then decide** (owner) |
| "Couple of hours" | not for full scope | not for full scope | **True only for the M1 batch once generators exist.** M2/M2b/M3b are days. |

Validation target (corrected by T0, 2026-11-24, [§4-validation](4-validation.md)): the 2023 slide-7 case
(inlet 0.075 mm, Re 12.2, x_r 224 vs 230 µm) is the **Karino & Goldsmith 1977 tubular sudden expansion**
(77.5 → 252 µm, E = 3.25, water + 1 % hardened RBCs ≈ Newtonian) as simulated by Wu et al. 2015 §3.1 — NOT Zhao 2008
(planar 100:200 µm, Hct 0.20, Re 8.3–41.7, no numeric x_r). → Validate with an **axisymmetric (wedge) 2D** case.
A planar E = 2 run at Re 12.2 gives x_r/h₁ = 0.54 (T0), consistent with planar physics, not comparable to the tube.

### 1. Nondimensional formulation (governs every run)
Scales: length h₁ (inlet height), velocity U = U_mean = Q/h₁ (per unit depth), viscosity μ_ref = μ∞,
stress μ_ref U/h₁. OpenFOAM runs in **star units** (ρ* = 1, h₁* = 1, U* = 1, ν* = μ_r/Re) — as `channel2d` already does.

| Group | Definition | Role | Range |
|---|---|---|---|
| Re | ρ U h₁ / μ∞ | inertia (matters in G2–G8, not in fully developed G1) | 0.5 – 25, log-uniform |
| Cu | λ U / h₁ | rheology time / flow time | from blood params × Re, see note |
| β | μ₀ / μ∞ | shear-thinning strength | 5 – 30 (`TODO: source` patient range) |
| n, a | Carreau exponents | shape | n 0.25–0.60, a = 2 (`TODO: source`) |
| φ | haematocrit | M2/M2b/M3 | 0.20 – 0.50 |
| τ*_lo, τ*_hi | (30, 60 Pa) · h₁/(μ∞U) | **shear window in star units** | derived per run |
| geometry | E = h₂/h₁, θ, throat ratio, R_c/h, L/h | design | per family (§2) |
| a/h₁ | RBC radius / channel | M2/M3 only | 0.03–0.11 |

Note — the dimensional window (30–60 Pa) breaks pure Re-similarity: two runs with equal (Re, Cu, β, n, geometry) give
identical star fields, but different τ*_lo. So one CFD run serves **many** window values in post-processing; the
surrogate input set is (Re, Cu, β, n, φ, geometry, τ*_lo). For one fixed blood, Cu and τ*_lo are tied:
Cu = λ τ_lo / (μ∞ τ*_lo).

### 2. Geometry families (h₁ = 1 in star units)
| ID | Family | Parameters (star units / ratios) | Mesher |
|---|---|---|---|
| G1 | straight | L/h ∈ [10, 30] — **analytic 1D for bulk**, 10–15 CFD anchors only | blockMesh |
| G2 | sudden expansion | E ∈ [1.5, 4], L_out/h₂ ≥ 10 | blockMesh |
| G3 | contraction–expansion (stenosis) | throat/h ∈ [0.3, 0.8], L_t/h ∈ [1, 5] | blockMesh |
| G4 | symmetric Y bifurcation | half-angle θ ∈ [20°, 90°], daughter/parent ∈ [0.5, 1] | blockMesh (Gmsh fallback) |
| G5 | asymmetric bifurcation | width ratio h_d1/h_d2 ∈ [0.5, 1], θ ∈ [30°, 90°] | blockMesh (Gmsh fallback) |
| G6 | serpentine | R_c/h ∈ [1.5, 6], 1–4 bends | blockMesh `arc` |
| G7 | network cell (expansion → bifurcation) | E × θ | composed blocks |

### 3. Corpus tiers (target ≈ 600 runs)
| Tier | Runs | Families | Purpose |
|---|---|---|---|
| GCI study | 10 cases × 3 meshes = 30 | 2 extreme corners of G2–G6 | fix mesh per family |
| M1 Newtonian + Carreau (+ Casson/Quemada once sourced) | 400 (≈ 64 Sobol pts × 6 families) | G2–G7 + G1 anchors | surrogate backbone |
| M2 mixture + Phillips (route a, pre-developed inlet φ) | 80 | G1, G2, G4 | Δ(M2 − M1) on CFL, window, HI |
| M2b multiphaseEulerFoam | 20 | G1, G2 (incl. Zhao) | continuum two-fluid anchor |
| M3a DPMFoam | ≤ 10 | G1, G2, φ ≤ 0.1 | qualitative per-cell dose |
| M3b 2D JAX LBM-IBM deformable | 8–10 | G1, G2 × 2–3 φ | calibrate M2 CFL (2D rings ≠ 3D biconcave) |

### 4. Storage (GCS, europe-west4)
`gs://<bucket>/corpus/v1/manifest.jsonl` (params, mesh, residuals, scalars, sha256, solver tag, cost) ·
`runs/<tier>/<family>/<run_id>.npz` (fields resampled to a fixed star-unit grid + mask; JSON params) ·
`gci/<family>/<case>.json` · `templates/<tier>/<family>.tar.gz`. Raw OpenFOAM directories are not kept.

### 5. Order of work and owners
| Step | Work | Owner | Gate |
|---|---|---|---|
| 0 | Source/verify constants; re-validate G2 at Zhao conditions | the team (lit) + the team (case) | numbers vs paper |
| 1 | Python blockMesh generator G2–G7 (star units), case templates | the team | mesh checks (checkMesh) |
| 2 | Pilot: 5 runs, measure s/run → VM choice | the team (VM) | cost estimate |
| 3 | GCI study (30 runs) | the team (VM) | GCI table |
| 4 | M1 batch (400) → bucket | the team + the team, one VM each | residuals, sanity plots |
| 5 | 2D post-processing (streamline / Eulerian D_I, fixed-grid resampling) | the team | tests vs G1 analytic |
| 6 | **M2b Euler–Euler** (`multiphaseEulerFoam`, stock closures from Wu et al. 2015 or the owner's 2023 COMSOL settings) — 20 anchors, expand if cheap | the team (VM) + the team (case) | CFL + x_r vs Zhao 2008 / 2023 Fig. 8 |
| 7 | M2 mixture + Phillips (custom solver, ~1 day) — optional; M3a ≤ 10; M3b 8–10 (own 2D LBM-IBM, days) | the team / the team | anchors |

Revision 2026-11-24 (owner): Euler–Euler with stock closures is a configuration task (hours), as in COMSOL — moved
ahead of the custom Phillips solver. Closures must still be named and cited in the article.
| 8 | Corpus report (figures, tables) → **write-up + scientific article draft** | the team drafts, the team reviews | sources, claims |

Stage-1 deliverable shown to the owner: after steps 0–5 (M1 corpus + GCI), then again after 6–7.

### 6. Revision v2 (owner, 2026-11-24): paired 4-variant corpus — supersedes M1b, M2, M3a in §3
Every **design** = (geometry family + its parameters, device scale h, U, haematocrit φ) is run in four paired
variants on the **same mesh** (GCI-chosen level), so every pair difference is a clean model-error label:

| Variant | Physics | Solver (ESI v2306) | RBCs |
|---|---|---|---|
| V1 `N` | Newtonian, μ = μ_eff(φ) | simpleFoam | none (uniform) |
| V2 `GN` | Carreau, parameters(φ) | simpleFoam + BirdCarreau | none (uniform) |
| V3 `EE-DL` | two-fluid, drag + Saffman-type lift, constant phase viscosities | reactingTwoPhaseEulerFoam, GidaspowErgunWenYu + LegendreMagnaudet | continuum φ |
| V4 `EE-full` | V3 + wall lubrication + virtual mass + shear-thinning RBC-phase viscosity | same, extra stock closures | continuum φ + cell-free layer |

- Haematocrit per design: 3 levels — anaemic ≈ 0.25, normal ≈ 0.45, polycythaemic ≈ 0.55 (`TODO: source` clinical
  ranges). V1/V2 need a φ-dependent blood viscosity / Carreau fit (`TODO: source`, the team literature task in T7).
- Common scale for all four variants: μ_ref = **plasma** viscosity μ_p (φ-independent), so Re_p = ρUh/μ_p and
  τ* = τ h/(μ_p U) are identical across variants and haematocrits of one design; the 30–60 Pa window in star units
  is per design. Every record also stores h, U, μ_p, d_RBC/h, t_ref = h/U.
- Device-space sampling (Sobol, seeded): h 25–150 µm, U 0.02–1 m/s (log), per family G2–G7, plus G1 anchors.
- Size: pilot 1 design × 3 φ × 4 variants per family (72 runs) → measure EE cost → then **8 designs/family**
  (48 × 3 × 4 = 576 runs, 288 Euler–Euler), extend to 16/family if cost allows.
- Kept: existing M1 (384, low-fidelity extra data), GCI, validation, M2b-A 20 runs (drag only, no lift = control).
- Dropped: M1b (replaced by V1/V2), M2 Phillips mixture, M3a DPMFoam (Lagrangian). M3b LBM-IBM parked (optional).
- Not modelled, stated in the article: shear-induced (Phillips) dispersion (not stock); wall lubrication is a
  bubbly-flow closure used as a stand-in for deformability-driven RBC wall lift.

#### 6.1 Amendments v2.1 (owner + the team/the team-Large review, 2026-11-24) — override §6 where they differ
- **Steady target.** Flow conditions are constant, so every V3/V4 result is a *steady* solution. The Euler–Euler solvers
  only exist as time-marching (PIMPLE) codes, so they march in pseudo-time to steady state: local time stepping
  (`ddtSchemes localEuler`) if the solver supports it (verify in source), otherwise large maxCo with a steady-state
  stop rule. Stop rule: relative change < 1e-4 over the last 10 % of iterations of Δp*, x_r, **near-wall φ
  (mean over the first 0.1 h₁ at the outlet) and CFL thickness** (outlet-mean φ alone is conserved, not a test).
- **V4** = V3 + wall lubrication + virtual mass, constant RBC-phase viscosity (shear-thinning RBC phase is not stock
  in v2306 → optional custom model later, FLAGged). Ablation on 6 anchors (1 per family): V3+wall-lub only,
  V3+virtual-mass only.
- **Drag** WenYu in V3/V4 (same as the M2b-A control); GidaspowErgunWenYu only for the Sezer comparison.
- **φ continuous**: Sobol over (geometry, h, U, φ ∈ [0.2, 0.6]) jointly, 24 designs per family (144 designs × 4
  variants = 576 runs), plus 3 designs per family pinned at clinical φ (anaemic / normal / polycythaemic, `TODO: source`).
- **Continuum limit**: d_RBC/h ≤ 0.1 → h ∈ [80, 150] µm for paired designs.
- **Developed inlet (owner choice)**: per design one short straight-channel (G1) precursor per Euler–Euler variant;
  its outlet profiles (α, U_plasma, U_rbc) are mapped onto the device inlet. Single-phase variants use the developed
  Newtonian / Carreau inlet profile.
- **Mesh**: no extra refinement for Euler–Euler. Reuse the M1 medium GCI mesh per family with wall grading so that
  ≥ 4 cells span the expected cell-free layer; confirm with a 3-level check on 2 Euler–Euler cases only.
- **Gates**: KG (φ = 0.01) = solver check only; physics gate = Sezer ΔP*(α_R) and Zhao depletion at φ 0.2–0.45.
- **Archives**: final time step only (+ mesh, dicts, 0/, logs).

#### 6.2 Amendment v2.2 (owner, 2026-11-24): Euler–Euler production in FEniCSx
- **V3/V4 production solver = own steady 2D two-fluid FEniCSx (dolfinx) code** by the team (`cfd/fenicsx/`), Newton
  solve with continuation in φ and Re, closures written explicitly: Wu 2015 drag with hindrance + Saffman lift
  ([§2-theory](2-theory.md)), V4 adds wall lift/lubrication, virtual mass (zero in steady flow → dropped, stated) and a
  shear-thinning RBC phase. Formulation reference: Wu 2015 + COMSOL "Euler–Euler Model, Laminar Flow" docs
  (`TODO: source` — owner to send 2023 model report or doc URL). Developed inlet from a FEniCSx G1 precursor.
- V1/V2 in OpenFOAM (validated) **and** in FEniCSx on the same mesh (cross-code check, cheap).
- **OpenFOAM Euler–Euler** (reactingTwoPhaseEulerFoam, stock closures) = gates (KG, Sezer, Zhao), the 20-run M2b-A
  control, and **at the end a random 10 % sample of the paired designs** (seeded) as a cross-code comparison.
- FEniCSx gates before production: Poiseuille φ → 0 limit vs analytic, KG x_r/d₁ (axisymmetric form), Zhao depletion,
  Sezer ΔP*(α_R) trend.

#### 6.3 Amendment v2.3 (owner, 2026-11-24): haematocrit range + carrier fluid ([§4-validation](4-validation.md) A13, A14)
- φ ∈ [0.01, 0.6]: Sobol over the full range **plus extra designs in the device regime 0.01–0.1** (Bernardelli 2025:
  1 % and 10 % Ht in PBS/GASP; Gracka 2022: 5 %). Suggested split per family: 12 designs φ ∈ [0.01, 0.1],
  12 designs φ ∈ [0.1, 0.6], + 3 pinned clinical designs.
- Carrier-fluid viscosity μ_c is a run parameter: plasma 0.96 mPa·s (whole blood) or buffer 1.1–1.7 mPa·s (Bernardelli
  Table 3). μ_ref = μ_c of that run.
- Shear window: under discussion (baseline 30–60 Pa + candidate 15–45 Pa variant; hemolysis always time-aware) —
  window is a post-processing input, the CFD runs do not depend on it.

### 7. CANONICAL RUN COUNTS (single source of truth; updated 2026-11-24 from campaign/paired_design.jsonl)
| Block | Runs | Solver | Status |
|---|---|---|---|
| Paired designs | **164** = 6 families (G2–G7) × 27 (12 φ 0.01–0.1, 12 φ 0.1–0.6, 3 clinical) + 2 G1 anchors | — | design file ready |
| V1 Newtonian | 164 | OpenFOAM | pilot 6/6 ok |
| V2 non-Newtonian (Carreau) | 164 | OpenFOAM | pilot 5/6 ok (G3 not converged) |
| V3 Euler–Euler, COMSOL 2023 model | 164 | FEniCSx | solver + pipeline ready, pilot 12/12 ok |
| V4 Euler–Euler, Wu/Sezer + wall lift | 164 | FEniCSx | converges (COMSOL recipe); wall lift pending |
| **Paired corpus total** | **656** (328 single-phase + **328 Euler–Euler**) | | |
| M1 legacy single-phase | 364 ok (+20 failed, labelled) | OpenFOAM | done, archived |
| GCI | 30 | OpenFOAM | done, archived |
| Validation (tube expansion) | 2 | OpenFOAM | done, archived |
| OpenFOAM Euler–Euler gates + M2b-A control | 8 gates + 20 control | OpenFOAM | partial (KG Re 37.8 pass) |
| Cross-code check | 10 % of 164 designs × 2 EE variants ≈ 32 | OpenFOAM | last |
Not runs: inlet profiles (computed inside each V3/V4 run).
