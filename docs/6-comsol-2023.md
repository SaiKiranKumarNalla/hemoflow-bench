# 6 · The 2023 COMSOL study (Nalla & Gereige)

The original Euler–Euler study that started this project.

## Contents

- [The 2023 COMSOL Euler–Euler study (Nalla & Gereige)](#the-2023-comsol-eulereuler-study-nalla--gereige)
- [The 2023 COMSOL two-phase model (as recovered from the slides)](#the-2023-comsol-two-phase-model-as-recovered-from-the-slides)

---

## The 2023 COMSOL Euler–Euler study (Nalla & Gereige)


Authors: **Sai Kiran Kumar Nalla & Elio Gereige**, MEC/BIO 586 "Biomechanics in Health and Disease",
January 2023 — report and 36-slide deck (`knowledge/papers/pdf/nalla-gereige-2023-report.pdf`,
`...-slides.pptx`). The owner confirmed the 2023 figures/results are his to share; **Gereige is credited
wherever they appear** (including `figures/fig09_comsol_2023.png`). Text below is our own summary; no figures
or equation images are copied.

### Setup (as recovered from slide 6 "Model Definition" and slide 7)

- COMSOL **"Euler–Euler Model, Laminar Flow"** (two-fluid, not the Mixture interface): continuous phase
  plasma, dispersed phase RBC modelled as **solid particles**, particle diameter $d_d = 8\,\mu$m.
- Drag: **Schiller–Naumann** (at particle Re ~ 1e-3 this is Stokes); mixture viscosity: **Krieger**;
  material properties taken from the Wu et al. 2015 OpenFOAM paper (their ref [8]). Material rows on the
  slide: ρ = 1038 kg/m³ with μ = 4×10⁻³ Pa·s and ρ = 1098 kg/m³ (unlabelled; 1098 is the usual RBC density,
  so the first row is most likely plasma — `TODO: confirm`).
- Inlet velocity 0.25 m/s (treated as the mean, $U = Q/A$; Re 3.37 in a 50 µm **circular tube** with the
  mixture viscosity); expansion validation case: inlet 0.075 mm (radius; $d_1 = 151\,\mu$m Karino–Goldsmith
  tube), Hct 0.22, Re 12.2. Walls: no-slip for both phases. Outlet: pressure $p_0 = 0$, no viscous stress.
- The model was run **time-dependent**; the cell-free layer was read from an **early snapshot**
  (owner recollection, 2026-11-24).

### Results (2023)

- Validation: reattachment point **224 µm vs 230 µm** literature on the expansion case; mesh-independence
  study on a 0.05 mm channel at inlet 0.25 m/s, Re 3.37; optimal encapsulation radius **0.56 R vs ≈ 0.6 R**
  (Piergiovanni 2020) in a straight channel.
- Encapsulation: high-index zone **0.2 < I < 0.4** with the index of Piergiovanni et al. 2020 Eq. 1
  ($I = v\tau\phi/(v_{max}\tau_{max}\phi_{max})$, see `docs/02_math.md` §3 caveats); optimal shear window
  **30–60 Pa**; optimal ring at ≈ 0.6 R covering **~57.8 % of the cross-section** (tubular pinch).
- Variable effects (slide 9): higher max shear at higher velocity / lower diameter; longer exposure at lower
  velocity / higher length (L/D fixed at 20); higher I at higher D and L; **higher velocities can stay below
  the hemolysis threshold in some geometries** — a limitation of the earlier literature, not of the physics.
- Extended-deformation (curved) geometry: D 50 µm at V = 0.332 m/s. The deck also covers drug transport and
  release (straight channel, symmetric/asymmetric bifurcations, 2D carotid, 3D aorta) — that is **Stage B,
  out of scope** for this benchmark.

### The lesson that motivated the current model hierarchy

The 2023 time-dependent runs showed a **cell-free layer only in early snapshots; the CFL disappeared later**,
while physical chips keep one. Our steady two-fluid solutions with drag + slip-based Saffman lift reproduce
the same failure (~1 % wall depletion at steady state, [§4-validation](4-validation.md) A6 update), and the owner's recollection
of the 2023 behaviour matches. Conclusion: **a steady CFL requires deformability-induced wall lift**
(Abkarian-type, force ∝ shear rate, independent of slip — `docs/02_math.md` §5, [§2-theory](2-theory.md)), which is
what variant V4 adds. The CFL is therefore visible **only transiently** in models without it — a concrete,
checkable modelling lesson carried into the v2 corpus design.

Figure: `figures/fig09_comsol_2023.png` (panel of the 2023 figures/results, credited Nalla & Gereige).

---

## The 2023 COMSOL two-phase model (as recovered from the slides)


Source: `knowledge/papers/pdf/nalla-gereige-2023-slides.pptx`, slide 6 "Model Definition" (COMSOL UI screenshots,
read 2026-11-24 by the team) and slide 7. The report (`…-report.pdf`) gives no settings; its ref [8] is the model's
literature basis.

| Item | Value on the slides | Note |
|---|---|---|
| Physics interface | **COMSOL "Euler-Euler Model, Laminar Flow"** | two-fluid, not the Mixture Model interface |
| Continuous phase | Plasma | |
| Dispersed phase | RBC, particle diameter d_d = **8 µm** | |
| Material rows shown | ρ = 1038 kg/m³ with μ = 4e-3 Pa·s; ρ = 1098 kg/m³ | which row belongs to which phase is not labelled on the slide; 1098 is the usual RBC density, so 1038 / 4 mPa·s is most likely the continuous phase — `TODO: confirm` |
| Inlet | u_c = u_c0, u_d = u_d0, φ_d = φ_d0 | inlet velocity 0.25 m/s (slide 7); φ_d0 not shown |
| Walls | no slip for both phases: u_c = u_d = 0 | |
| Outlet | "Pressure, no viscous stress", p₀ = 0; τ_c·n = τ_d·n = 0 | |
| Validation cases | 50 µm channel, Re 3.37 (mesh study); sudden expansion inlet 75 µm, Hct 0.22, Re 12.2, reattachment 224 µm vs 230 µm literature | slide 7 |

**Not on the slides:** drag law, dispersed-phase viscosity model, lift, turbulent/collision dispersion, maximum
packing. **Owner confirmed (2026-11-24): the 2023 model was based on and validated against Wu et al. 2015** —
so its closures are the reference closures for M2b:

> Wu, W.-T., Yang, F., Antaki, J. F., Aubry, N., Massoudi, M. (2015). Study of blood flow in several benchmark
> micro-channels using a two-fluid approach. *Int. J. Eng. Sci.* 95, 49–59. doi:10.1016/j.ijengsci.2015.06.004

→ For M2b, take the closures from Wu et al. 2015 (being verified in task T0, [§4-validation](4-validation.md))
and use the slide-6 boundary conditions and phase properties above. Run in star units (h₁ = 1, U = 1).

### Owner confirmation 2026-11-24 (2023 COMSOL settings)
Dispersed phase type **solid particles**; drag **Schiller–Naumann** (owner: "Schiller–Naumann or no drag model" — COMSOL
Euler–Euler always includes drag; at Re_p ~ 1e-3 Schiller–Naumann = Stokes); mixture viscosity **Krieger**; material
properties "taken exactly from the OpenFOAM paper" (Wu 2015). Note: Wu 2015 itself used Stokes drag + Rusche–Issa
hindrance and a Yeleswarapu-type RBC viscosity, not Krieger — so the 2023 model = COMSOL built-in closures + Wu
properties. Krieger φ_max used in 2023: unknown (COMSOL default 0.62 for solids). Implemented as V3
`closure="comsol-solid"` in cfd/fenicsx/twofluid.py.
**Decision (owner, 2026-11-24):** V3 uses Krieger with an RBC packing limit φ_max = 2/k∞ = 0.966 (Quemada high-shear
k∞ = 2.07, [§2-theory](2-theory.md)) instead of COMSOL's rigid-sphere default 0.62 (which gives μ/μ_p = 7.4 at φ 0.45 and 205 at
0.60); one comparison run at 0.62 for the article.
**Owner recollection (2026-11-24):** in 2023 the COMSOL Euler–Euler model was run **time-dependent** and the cell-free
layer was read from an **early snapshot**; the CFL disappeared later in the run, unlike in physical chips where it
persists. Consistent with our steady V4 result (Zhao, COMSOL recipe: wall/core φ 0.1987/0.2012) and the OpenFOAM EE
gates: drag + Saffman lift alone cannot sustain a CFL for near-neutrally-buoyant cells.

### 2023 report + slides read
Report "Shear encapsulation of RBCs for drug delivery" (MEC/BIO 586, 24 Jan 2023, Nalla & Gereige) + 36 slides.
- The **encapsulation index I (Eq. 1)** and the **optimal shear window 30–60 Pa** and **0.2 < I < 0.4** are taken from
  **Piergiovanni, Casagrande, Taverna et al., Ann Biomed Eng 48:236–246 (2020), doi 10.1007/s10439-019-02342-w** (report
  ref. [2]), which used a COMSOL mixture model; slides 3–4 cite the same paper ("optimal zone 57.8 % of the cross-section,
  0.6R, tubular pinch effect"). → [§4-validation](4-validation.md) A7/A8 primary source = Piergiovanni 2020 (`secondary` until read).
- Validation in 2023: sudden expansion reattachment 224 µm vs 230 µm literature (KG, Re 12.2); "Formation of Cell Free
  Layers near the wall" shown on the KG case at inlet Ht 0.22 (slide 7); Zhao 2008 φ comparison via Wu 2015 Fig. (report Fig. 8).
- Hemolysis limit reference in 2023: Sohrabi & Liu 2017, Artif Organs 41:E80, doi 10.1111/aor.12832.
- Initial condition and CFL snapshot time are **not stated** in the report or slide text → filling-front hypothesis to be
  tested by a V3-transient reproduction run.
