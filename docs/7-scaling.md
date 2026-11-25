# 7 · The data-scaling bottleneck

Cost of high-fidelity data and why surrogates are needed.

## Contents

- [Data scaling: why this benchmark needs surrogates](#data-scaling-why-this-benchmark-needs-surrogates)

---

## Data scaling: why this benchmark needs surrogates


The portfolio question is a **data-scaling bottleneck**: high-fidelity CFD of RBC suspensions is too
expensive to sample the joint design × patient space, so the benchmark pairs a cheap, validated,
nondimensional 2D corpus with multi-fidelity surrogates (PINNs / neural operators / GPs).

### Measured cost of the legacy 2D runs (OpenFOAM v2306, star units)

All numbers from `data/corpus_v1_legacy/manifest.jsonl` (`wall_seconds`, `n_cells`) and (internal lab log).
Runs executed on 8-vCPU spot VMs, one case per core.

| Quantity | Value |
|---|---|
| Pilot point (30 400 cells × 2500 it) | 151.1 s (Carreau) / 231.5 s (Newtonian) wall on a 2-vCPU VM → **~0.06–0.09 s per 1000 cell-iterations** |
| M1 corpus, 366 ok runs: wall time per run | median **16.8 s**, mean 53 s, min 3.9 s, max 1107 s; total ≈ **5.4 core-hours** |
| — per family (median s/run) | G2 105 (mean 195; up to ~81 k cells), G7 46, G3 18, G6 18.5, G4 7.3, G5 6.3 |
| Whole 384-run batch (mesh + solve + extract + archive, 8 cores) | ≈ 1.0 h wall ≈ 8.2 core-h; GCI study (30 runs) 4.5 min wall; KG validation (2 runs, 28 k cells) 4 min wall |
| Total cloud cost of the entire legacy campaign (validation + GCI + corpus + retries) | ≈ **$0.19** on spot VMs (logged in (internal lab log)) |

So a 2D single-phase run costs **seconds to ~18 minutes** (median ~17 s) at 7 k–80 k cells. Figure:
`figures/fig10_data_scaling.png` (cost per run vs cell count).

### 2D → 3D extrapolation (estimates, labelled as estimates)

- A 3D extrusion of the level-2 meshes at comparable resolution multiplies the cell count by roughly the
  depth in cells (~4–10× for square channels) → **~10⁵–10⁶ cells per 3D run**; at roughly linear cost in
  cells at fixed iteration count that is **minutes to tens of minutes per 3D single-phase run on CPU**
  (*estimate*, not measured).
- A 3D **two-fluid** run is worse: the time-marching Euler–Euler codes need pseudo-time to steady state —
  measured on the order of **~1 h per level-2 expansion case** for the gate runs (*measured on the
  in-progress v2 work*, quoted only as a cost datapoint).
- Cell-resolved 3D (LBM + immersed boundary, HemoCell-class, arXiv:2305.02752) costs **hours per run**
  ([§3-methods](3-methods.md), L3).
- The design × patient space is large: the in-progress v2 plan samples on the order of 10² designs × 3
  haematocrit levels × 4 model variants, and the surrogate must extrapolate across the continuous
  (Re, Cu, β, n, φ, geometry) space — i.e. **10³–10⁴ CFD evaluations** for a dense grid. At 2D cost that is
  days; at 3D cost, years. This is the bottleneck the benchmark is built around.

### Why surrogates / PINNs / operators — with precedents

- A NN surrogate for the power-law hemolysis model reached ~10 % error on the FDA nozzle benchmark (COMSOL
  application paper 136422, listed in [§8-literature](8-literature.md)).
- Neural operators in blood: DeepONet for platelet deformation in shear (arXiv:2503.12074); physics-guided
  surrogate for DLD device design (arXiv:2512.05649); multi-fidelity FNO with transfer learning and
  multi-fidelity scaling laws (arXiv:2304.06972, arXiv:2608.04708, arXiv:2511.01830); multi-scale RBC trauma
  coupling continuum CFD with cell-level damage (arXiv:2411.13002). All listed with links in
  [§8-literature](8-literature.md).
- Our own measured datapoint on legacy data: a physics-only PINN (soft BCs, MLP 64×4) on the lowest-Re
  Newtonian G2 run `M1_G2_052` (Re 1.017) reached only **85.97 % rel-L2 error in u** — diagnosed as a mass-flux
  leak local minimum, not a formulation error; the recommended fix is a divergence-free streamfunction ansatz
  (T9 report in (internal planning notes)). Lesson encoded in the benchmark: PINN training needs the
  divergence-free/integral constraint, and CFD fields remain the reference.

### The multi-fidelity plan

| Level | Model | Cost | Role |
|---|---|---|---|
| L0 | analytic (Poiseuille, Fåhræus, CFL estimate) | µs | sanity checks, features, initial guesses |
| L1 | 2D single-phase non-Newtonian (this legacy corpus) + 2D two-fluid | s–min/run | bulk training data |
| L2 | 3D lattice Boltzmann (JAX, GPU), ~30–80 s/run on A100 | ~100–300 runs | high-fidelity set, verification |
| L3 | cell-resolved LBM-IBM (HemoCell-class) | hours/run | calibrate thresholds (10–20 runs) |

Surrogates: **S0** GP/MLP ensemble on scalars (mean I, f_window, HI, Δp + uncertainty); **S1** FNO/U-Net on
fields (u, v, τ, φ, I) from L1; **S2** = S1 pretrained on L1, fine-tuned on L2 (multi-fidelity). Inverse
design: optimise (geometry, flow) per synthetic patient vector on S0/S2 under the HI constraint, Monte-Carlo
uncertainty propagation (MC-2), verify top designs with L2 ([§3-methods](3-methods.md)).

The paired four-variant corpus now **in progress** (V1 Newtonian, V2 Carreau, V3 Euler–Euler drag+lift,
V4 Euler–Euler full — same mesh per design, common plasma-viscosity scale, steady FEniCSx two-fluid
production solver with OpenFOAM gates) is exactly the L1 bulk-data layer of this plan; **no v2 results are
claimed in this portfolio**.
