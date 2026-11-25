# 5 · Data: the legacy corpus

Data card of the legacy single-phase corpus and the figure index.

## Contents

- [Data card: the legacy M1 single-phase corpus (v1)](#data-card-the-legacy-m1-single-phase-corpus-v1)
- [Figure captions](#figure-captions)
- [Legacy figure index](#legacy-figure-index)

---

## Data card: the legacy M1 single-phase corpus (v1)


Source: `data/corpus_v1_legacy/` (`manifest.jsonl` + `m1/<family>/<run_id>.npz`, README there). Solver:
OpenFOAM ESI v2306 `simpleFoam`, steady, incompressible, laminar, 2D planar, **star units**
($h_1^* = 1$, $U^* = 1$, $\mu_\infty^* = 1$, $\rho^* = 1$, $\nu^* = \mu_r/\mathrm{Re}$; viscous-scaled pressure
$p^* = \mathrm{Re}\,p_{OF}$). Sampling: scrambled Sobol, seed 20261124, 64 points per family.

### Counts and coverage

| Item | Value |
|---|---|
| Runs designed / executed | **384** (64 per family × 6 families G2–G7) |
| `status == "ok"` (usable) | **366** (95.3 %); all `converged_by: "residual"` (initial residuals < 1e-5) |
| `failed` (`not_converged`) | **18** — all BirdCarreau, concentrated in G3 (14), G5 (3), G6 (1); 0/36 Newtonian failed. Retried at 4× iterations: not iteration-starved (residual floor); kept out of training/testing |
| Rheology (ok runs) | 36 Newtonian, 330 BirdCarreau |
| Re | 0.505 – 24.98 |
| Cu = λU/h₁ | 1.01 – 9.56×10⁴ (206/330 ≈ 62 % of Carreau runs have Cu < 10³ — see caveats) |
| β = μ₀/μ∞ | 5.0 – 29.9 (design bounds 5–30, `TODO: source` patient range) |
| n, a | 0.250 – 0.598, a = 2 (bounds `TODO: source`) |
| Mesh | GCI level 2 for all families (cells/run: 6 624 – 79 704, median 15 390; G2 up to 81 k) |
| Stored scalars per run | Δp* ∈ [68.9, 4729.7]; x_r/h₁ ∈ [0, 2.83] (178 expansion runs); f_window ∈ [0, 0.402] (stored with the fixed 2023 window — see caveats) |

Geometry families (star units, from [3-methods](3-methods.md) §2): **G2** sudden expansion
($E = h_2/h_1 \in [1.5, 4]$), **G3** contraction–expansion/stenosis (throat/h₁ ∈ [0.3, 0.8],
$L_t/h_1 \in [1, 5]$), **G4** symmetric Y bifurcation (half-angle θ ∈ [20°, 90°], daughter/parent ∈ [0.5, 1]),
**G5** asymmetric bifurcation (width ratio ∈ [0.5, 1]), **G6** serpentine ($R_c/h_1 \in [1.5, 6]$, 1–4 bends),
**G7** network cell (expansion → bifurcation). Plus the **KG** Karino–Goldsmith axisymmetric tube
($d_1 = 151\,\mu$m, E = 3.3377) used only for validation. Figure: `figures/fig01_geometries.png`;
coverage scatter: `figures/fig05_parameter_coverage.png`.

### NPZ schema (one file per run)

| Key | Shape | Meaning |
|---|---|---|
| `x_star, y_star, cell_volumes` | (N,) | cell centres and areas of the OpenFOAM mesh (unstructured cloud) |
| `U_star` | (N,3) | velocity (u, v, 0) |
| `p_star` | (N,) | viscous-scaled pressure (outlet p = 0) |
| `gdot_star, tau_star` | (N,) | shear rate and shear stress |
| `grid_x, grid_y, grid_mask` | (256,), (96,), (96,256) | regular grid + fluid mask |
| `grid_ux, grid_uy, grid_p, grid_gdot, grid_tau` | (96,256) | fields resampled on the grid (NaN/0 outside mask) |
| `outlet_*` | (100,) | outlet profiles (s, normal velocity, γ̇, τ, weights, branch id) |
| `params` | JSON string | run parameters, convergence info, scalars (Δp*, x_r, …) |

Loaders must **de-duplicate the append-only manifest by `run_id`, taking the LAST row** (re-runs append rows;
`status: ok` wins). Figures from this corpus: `figures/fig02_velocity_fields.png`,
`figures/fig03_shear_fields.png`, `figures/fig04_profiles.png`, `figures/fig08_window_fraction.png`,
`animations/anim01_particles_<family>.gif`, `animations/anim02_re_sweep_G2.gif`,
`animations/anim03_carreau_vs_newtonian.gif`.

### Physics-only training (PINN/PINO): the exact boundary-value problem

Use the CFD fields as reference solutions only; the network needs PDE + geometry + BCs:

- PDE (star units): $\nabla \cdot u = 0$; $\mathrm{Re}(u\cdot\nabla)u = -\nabla p^* + \nabla\cdot[2\mu^*(\dot\gamma)D(u)]$
  with $\mu^* = 1$ (Newtonian) or the Carreau law of `docs/02_math.md` §2; $\dot\gamma^* = \sqrt{2D:D}$
  (check `params.rheology`).
- Inlet: Newtonian → parabolic mean-1 profile $6y(1-y)$; Carreau → uniform $u = (1,0)$. Walls: no-slip.
  Outlet(s): $p = 0$, zero normal gradient of u (all outlets at bifurcations: $p = 0$).
- Geometry: rebuild the exact boundary from `params["geometry"]` with the repo's geometry builder
  (`cfd/gen/geometry.py`), then sample boundary/collocation points. Steady, laminar, Re 0.5–25.

### Caveats (must be read before using the corpus)

1. **~62 % of the Carreau runs have Cu < 10³**, which under the consistent dimensional mapping corresponds to
   **mm-scale channels** — valid flow fields, but outside the 50–150 µm encapsulation-device regime
   ([9-decisions](9-decisions.md), "consistent dimensional mapping").
2. **`tau_window_star` / stored `f_window` used the fixed 2023 star window** — do not use as-is; recompute any
   window per run from the stored outlet profiles / fields (the dimensional window 30–60 Pa breaks pure
   Re-similarity: one CFD run serves many window values in post-processing).
3. **Max wall τ is singular at sharp corners** (rises monotonically with mesh refinement in 9/10 GCI corners,
   e.g. 81.6 → 97.4 → 115.9) — dropped as a target; use surface-resolved metrics instead
   (`cfd/openfoam/gci_summary.md`).
4. **GCI flags**: level 2 chosen for all families; $x_r/h_1$ is not 2 %-certified for β→30, n→0.25 in G2/G3
   (GCI 3.2–3.9 %); G5's literal level-1 pass was an artefact (direct L1→L2 change of f_window 11.4 %);
   G7 has no GCI case (level 2 by fallback). Δp* grid-converged within 0.9 %.
5. All 366 ok runs are `converged_by: "residual"`; the QoI-drift criterion never fired — stated in the card
   so nobody over-reads the convergence label.
6. 2D planar only: real devices are square/rectangular channels (Bernardelli 50×50 µm, Sezer 100×100 µm) —
   absolute numbers carry a 2D↔3D factor `TODO` ([§4-validation](4-validation.md) A12).

---

## Figure captions


- **Figure 1.** Fluid masks for six representative legacy Newtonian runs and the Karino–Goldsmith tube outline. The six run IDs are listed in the index.
- **Figure 2.** Nondimensional speed in the same six Newtonian runs. These are steady velocity fields.
- **Figure 3.** Nondimensional shear stress in the same six runs, with available 30 and 60 Pa contours under the displayed physical scaling.
- **Figure 4.** Not yet produced: comparable Newtonian and Carreau inlet and downstream profiles are required.
- **Figure 5.** Legacy design coverage from 384 latest M1 manifest records: 366 successful and 18 failed.
- **Figure 6.** Reported L1→L2 GCI for pressure drop and reattachment, where available, across ten GCI cases. The maximum plotted value is 3.18%; the source specifies a 2% target for these quantities, while 5% is shown as the requested reference.
- **Figure 7.** Karino–Goldsmith reattachment comparison at Re 12.2 and 37.8, with Wu 2015 targets 1.5232 and 4.7351 and legacy results 1.5386 and 4.6615.
- **Figure 8.** Outlet flux-weighted 30–60 Pa window fraction for the 366 successful legacy M1 runs. It is a screening measure, not particle exposure or encapsulation, and depends on the stated physical mappings.
- **Figure 9.** Selected 2023 COMSOL results from the owner-provided presentation by Nalla & Gereige. The panels reproduce the original presentation images and require scientific review before publication.
- **Figure 10.** Observed wall time versus cells for 366 successful legacy M1 runs; the median is 16.8031 s. The 50–100× 3D band is an illustrative task-specified scenario, not a measured scaling law.

---

## Legacy figure index


Figures use the latest manifest row per M1 run, and include only `ok` runs except the design-coverage plot. Star fields use h₁, inlet mean U, and μ∞ as reference scales. Figures are preliminary and require review.

- `fig01_geometries.png` — fluid masks for M1_G2_000, M1_G3_000, M1_G4_000, M1_G5_000, M1_G6_000, M1_G7_000 and KG tube outline; `tools/portfolio/legacy_figures.py`.
- `fig02_velocity_fields.png` — speed fields for the six runs above; `tools/portfolio/legacy_figures.py`.
- `fig03_shear_fields.png` — shear fields and 30/60 Pa reference contours for the six runs above; Newtonian physical mapping is stated on the figure; `tools/portfolio/legacy_figures.py`.
- `fig05_parameter_coverage.png` — all 384 latest M1 rows, 366 ok and 18 failed; `tools/portfolio/legacy_figures.py`. Markers in order G2–G7 are circle, square, up triangle, down triangle, diamond, filled plus.
- `fig06_gci.png` — GCI_G2 through GCI_G6, inertial and shearthin; fine-grid GCI estimates, with unavailable xᵣ omitted; `tools/portfolio/gci.py`.
- `fig07_validation_kg.png` — KG_Wu2015_Re12p2_L1 and KG_Wu2015_Re37p8_L1 compared with Wu 2015 targets; `tools/portfolio/legacy_figures.py`.
- `fig08_window_fraction.png` — all 366 ok M1 runs, recomputed from outlet profiles with stated reference scales; screening only, not exposure or encapsulation; `tools/portfolio/legacy_figures.py`.
- `fig10_data_scaling.png` — measured wall times and cell counts for 366 ok M1 runs, with an illustrative 3D estimate; `tools/portfolio/scaling.py`.

### T13b candidates

- `../animations/anim01_particles_G2.gif` and `.mp4` — M1_G2_000; `tools/portfolio/animations.py`.
- `../animations/anim01_particles_G3.gif` and `.mp4` — M1_G3_000; `tools/portfolio/animations.py`.
- `../animations/anim01_particles_G4.gif` and `.mp4` — M1_G4_000; `tools/portfolio/animations.py`.
- `../animations/anim01_particles_G5.gif` and `.mp4` — M1_G5_000; `tools/portfolio/animations.py`.
- `../animations/anim01_particles_G6.gif` and `.mp4` — M1_G6_000; `tools/portfolio/animations.py`.
- `../animations/anim01_particles_G7.gif` and `.mp4` — M1_G7_000; `tools/portfolio/animations.py`.
- `../animations/anim02_re_sweep_G2.gif` and `.mp4` — M1_G2_052, M1_G2_000, M1_G2_031, M1_G2_021, M1_G2_042, M1_G2_010; geometry varies; `tools/portfolio/re_sweep.py`.
- `fig06_gci.png` and `.svg` — ten legacy GCI cases, reported L1→L2 values; `tools/portfolio/gci.py`.
- `fig09_comsol_2023.png` and `.svg` — 2023 slides, Nalla & Gereige; `tools/portfolio/comsol_2023.py`.
- `fig10_data_scaling.png` and `.svg` — all 366 ok M1 runs; `tools/portfolio/scaling.py`.

These are candidates, not approved publication figures; remaining first-pass entries above have not been restyled.
