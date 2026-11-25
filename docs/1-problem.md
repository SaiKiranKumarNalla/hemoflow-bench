# 1 · The problem: RBC drug carriers and shear

Why red blood cells, why shear, and the research question.

## Contents

- [The biology problem: RBCs as drug carriers under controlled shear](#the-biology-problem-rbcs-as-drug-carriers-under-controlled-shear)
- [Background](#background)
- [Research question](#research-question)

---

## The biology problem: RBCs as drug carriers under controlled shear


### Why RBCs

Red blood cells make attractive drug carriers: they are ~45 % of blood volume, highly biocompatible (no
PEGylation needed to avoid immune clearance), have a high surface-to-volume ratio (they lack a nucleus), and
circulate for months — a drug loaded inside an RBC acts as a slow-release depot, analogous to a slow
infusion with much larger capacity (Glassman et al. 2021, doi:10.1016/j.addr.2021.113992; Koleva et al. 2020,
doi:10.3390/pharmaceutics12030276; both `secondary` as listed in [§8-literature](8-literature.md)).

Loading methods include osmotic swelling, electroporation, ultrasound and **shear-induced encapsulation**.
The first three often damage cells (hemolysis, permanent shape change — the 2023 report of Nalla & Gereige
reproduces the dead-cell percentages of electroporation, report Fig. 3); controlled shear can open membrane
pores **transiently**, letting cargo diffuse in without lasting damage (Piergiovanni et al. 2020,
doi:10.1007/s10439-019-02342-w).

### The shear mechanism and its two-sided risk

- Above a shear-stress threshold, applied for long enough, membrane pores open temporarily and cargo
  (e.g. FITC-dextran 40 kDa in Piergiovanni et al. 2020) diffuses in. Bernardelli et al. 2025
  (doi:10.1007/s10439-025-03678-2, PMC12006216, read in full → `verified`) achieved loading of **70–94 % of
  cells** in a 50 µm square microchannel at a maximum wall shear of **49.7 Pa** (50 µL/min, 1 % Hct), i.e.
  sub-hemolytic.
- Too much stress, or too long an exposure, causes **hemolysis**. Damage is a **dose × time** effect, classically
  the power law $HI = C\,\tau^{\alpha}\,t^{\beta}$ (Giersiepen et al. 1990; constants `secondary`, see
  `docs/02_math.md` §4). Sohrabi & Liu 2017 (doi:10.1111/aor.12832) give the cellular basis of
  shear-induced hemolysis.
- Shear is also the **release** trigger once the cells are in the bloodstream: Amar et al. 2026 (Small,
  doi:10.1002/smll.74818, PMC13580257, read in full → `verified`) report transient pores at ≈ **15 Pa** and a
  hemolysis threshold around ≈ **45 Pa** (static), with release correlating with membrane areal strain
  (pores above ~4 % areal strain, Evans 1976 — `secondary`). Stage B (release in arteries) is out of scope
  here, but the thresholds matter for the safety constraint in Stage A.

### Thresholds used in this benchmark (with labels)

| Quantity | Value | Label | Source |
|---|---|---|---|
| Encapsulation shear window (2023 work) | **30–60 Pa** | from the 2023 project, based on literature | Nalla & Gereige 2023 report/slides |
| Pore-opening threshold | ≈ 15 Pa | `secondary` (Amar 2026 intro, citing [18,19]) | [§4-validation](4-validation.md) A7 |
| Hemolysis threshold (static) | ≈ 45 Pa | `secondary` (same) | [§4-validation](4-validation.md) A7 |
| Sub-hemolytic loading point | 49.7 Pa × ~0.18 s transit → HI ≈ 0.12 % | `verified` (Bernardelli 2025, transit time computed by us) | [§4-validation](4-validation.md) A9 |
| Window is **time-aware** | 45 Pa reaches HI 0.5 % only after ≈ 1.5 s | derived from Giersiepen constants (`secondary`) | [§4-validation](4-validation.md) window decision |

**Flag (open):** the lower bound of the window may be ~15 Pa rather than 30 Pa, and the upper bound depends on
exposure time — this is why the benchmark computes time-aware dose metrics in addition to static window
fractions (`docs/02_math.md` §3, [§4-validation](4-validation.md)).

### Why geometry and flow rate matter

In straight channels RBCs migrate away from the walls (the **cell-free layer**, CFL) and gather in a ring
(tubular pinch): near the wall the shear is high but few cells and low velocity; at the centre the velocity is
high but shear is low. The 2023 COMSOL study (Nalla & Gereige) found the optimal encapsulation ring at
≈ 0.56–0.6 R covering ~57.8 % of the cross-section, a high-index zone $0.2 < I < 0.4$, and showed that
higher velocities can stay below the hemolysis threshold in some geometries — the encapsulation vs hemolysis
trade-off is geometry-dependent. See `docs/05_comsol_2023.md`.

### The patient dimension

Encapsulation efficiency depends on the blood itself: haematocrit and albumin change loading (Bernardelli et
al. 2025, `verified`; device suspensions run at 1–10 % Hct in PBS/GASP buffers of 1.12–1.68 mPa·s, Table 3),
and RBC deformability (elongation index, MCV) scales the pore-opening and hemolysis thresholds
(`TODO: source` ranges, [§2-theory](2-theory.md)). Hence the benchmark treats the blood as a **synthetic patient
vector** (Hct, plasma viscosity, albumin, EI, MCV, fibrinogen — ranges in [§2-theory](2-theory.md)) sampled from
published ranges. No real patient data is used, ever.

### Problem statement

> For a given (synthetic) patient blood vector, which microfluidic geometry and flow rate maximise mean RBC
> encapsulation while keeping hemolysis below a safe threshold, how does the optimum shift between patients,
> and how robust is it to measurement uncertainty?

This is what the surrogate in `hemoflow-bench` is built to answer — see `docs/06_data_scaling.md` for why a
surrogate is needed at all (the data-scaling bottleneck) and `docs/02_math.md` for the exact metrics.

---

## Background


### RBCs as drug carriers
RBCs are ~45 % of blood volume, biocompatible, circulate for months and have a high surface-to-volume ratio, so a
drug loaded inside them can act as a slow-release depot (Glassman et al. 2021; Koleva et al. 2020).
Loading methods: osmotic swelling, electroporation, ultrasound, and **shear-induced encapsulation**. The first three
often damage cells (hemolysis, shape change); controlled shear can open membrane pores transiently without lasting
damage (Piergiovanni et al. 2020).

### Shear-induced encapsulation (mechanism)
- Above a shear-stress threshold applied long enough, membrane pores open temporarily and cargo diffuses in.
- Too much stress or exposure → hemolysis (Sohrabi & Liu 2017; power-law damage models, see 04).
- Efficiency depends on flow velocity, local shear, exposure time and haematocrit. Piergiovanni et al. define a
  normalised **encapsulation index I** (their Eq. 1). `TODO: transcribe Eq. 1 exactly from the paper before coding.`
- In straight channels RBCs migrate away from the wall (cell-free layer) and gather in a ring (tubular pinch):
  near the wall shear is high but few cells/low velocity; at the centre velocity is high but shear is low.

### What the 2023 project found (Nalla & Gereige, MEC/BIO 586, report + 36-slide deck)
- Model: Euler–Euler / mixture two-phase blood (plasma + RBC), non-Newtonian, in COMSOL.
- Validation: sudden-expansion reattachment **224 µm** vs **230 µm** literature; mesh independence (0.05 mm channel,
  inlet 0.25 m/s, Re 3.37; also inlet 0.075 mm, Hct 0.22, Re 12.2); optimal encapsulation radius **0.56R** vs ≈0.6R.
- Encapsulation: high-I zone 0.2 < I < 0.4; optimal shear ≈ **30–60 Pa**; optimal ring ≈0.6R covering ~57.8 % of
  the cross-section; higher max shear at higher V / lower D; longer exposure at lower V / higher L; higher I at higher
  D and L (L/D fixed 20). Showed higher velocities can stay below the hemolysis threshold in some geometries.
- Curved geometry ("extended deformation"): D 50 µm, V 0.332 m/s.
- (Stage B — drug release in bifurcations, carotid, 3D aorta — exists but is out of scope here.)
- Data: COMSOL files may or may not still exist (owner checking). If found, use as extra validation points.

---

## Research question


> For a given patient blood vector, which microfluidic geometry (type and dimensions) and flow rate maximise mean
> RBC encapsulation while keeping hemolysis below a safe threshold, how does the optimum shift between patients,
> and how robust is it to measurement uncertainty?

Addresses gaps 1–4 in `02-literature.md`. Deliverables:
1. Validated 2D mixture solver (L1) and a small validated 3D LBM (L2).
2. A dataset of thousands of L1 runs and ~100–300 L2 runs.
3. Surrogates S0 (scalar), S1 (2D fields), S2 (3D-corrected).
4. Robust inverse design per patient (Monte Carlo uncertainty propagation), verified with L2.
5. Honest limitations: 2D vs 3D, empirical index I, no experimental validation of new designs.
