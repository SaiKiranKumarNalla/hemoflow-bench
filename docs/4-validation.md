# 4 · Validation, constants and assumptions

Replicated papers, verified constants, the assumption register and grid convergence.

## Contents

- [Validation: what we replicated, against which papers, with which numbers](#validation-what-we-replicated-against-which-papers-with-which-numbers)
- [Validation targets (must pass before data generation)](#validation-targets-must-pass-before-data-generation)
- [Constants verification (task T0, 2026-11-24, the team)](#constants-verification-task-t0-2026-11-24-the-team)
- [Assumption register (math ↔ biology/experiment) and validation cases](#assumption-register-math--biologyexperiment-and-validation-cases)
- [GCI study — 2D M1 mesh independence (T3, the team, 2026-11-24)](#gci-study--2d-m1-mesh-independence-t3-the-team-2026-11-24)
- [Paired-design GCI study — 2D V1/V2 mesh independence at Re ≤ 60, φ ≤ 0.6, τ_w ≤ 150 Pa (T10 Part 3, the team, 2026-11-24)](#paired-design-gci-study--2d-v1v2-mesh-independence-at-re--60---06-w--150-pa-t10-part-3-the-team-2026-11-24)

---

## Validation: what we replicated, against which papers, with which numbers


Rule: a solver earns the right to generate training data only after it reproduces published benchmark numbers.
Targets are from [§4-validation](4-validation.md); statuses from (internal lab log) and [§4-validation](4-validation.md).
Labels: `verified-primary` = original full text read; `secondary` / `abstract` as marked in `knowledge/`.

### Reproduced (legacy 2D OpenFOAM runs)

| Case | Quantity | Target | Ours | Deviation | Status |
|---|---|---|---|---|---|
| Plane Poiseuille channel (G1, Newtonian, level-2 mesh) | outlet velocity profile vs analytic $u = 6U y(1-y)$ | rel. L2 < 2 % | **0.7271 %**; centre-line $u_{max} = 0.9925$ vs 1.0; $p(x=9) = 7.9602$ vs 8.0 (0.50 %) | — | **PASS** (phase P1, (internal lab log)) |
| Karino–Goldsmith tubular sudden expansion (axisymmetric wedge, $d_1 = 151\,\mu$m, E = 3.3377, φ = 0.01), Re 12.2 | reattachment $x_r/d_1$ | **1.5232** (230 µm; Wu 2015 Fig. 3, `verified-primary` PMC4521229) | **1.5386** | **+1.01 %** | **PASS** (gate was 15 %) |
| Same, Re 37.8 | $x_r/d_1$ | **4.7351** (715 µm) | **4.6615** | **−1.55 %** | **PASS** |

These are the 2 legacy validation runs (`validation` tier, 28 008 cells each, both `converged_by: residual`).
Figure: `figures/fig07_validation_kg.png`. Geometry note: the 2023 deck's "inlet 0.075 mm, Re 12.2,
$x_r$ 224 vs 230 µm" case is this Karino–Goldsmith **tube** (0.0755 mm is the *radius*; Wu 2015 Re is
diameter-based), not Zhao 2008 — corrected in [§4-validation](4-validation.md) flag 1. Our 2023 COMSOL model reproduced
$x_r = 224\,\mu$m vs the 230 µm literature value on that case (Nalla & Gereige, slide 7).

### Used as validation targets (papers, what they provide, our status)

| Paper | What it provides | How we use it | Status |
|---|---|---|---|
| **Wu et al. 2015**, Int J Eng Sci 95:49–59, doi:10.1016/j.ijengsci.2015.06.004, PMC4521229, PMID 26240438 (`verified-primary`, read in full) | Two-fluid OpenFOAM benchmark cases: KG tube (above), Patrick channel (Hct 0.45, RBC velocity profiles at z = 8/16 µm), Zhao expansion (Hct 0.20, 125 544 nodes), crevices (Hct 0.40); closures transcribed in [§2-theory](2-theory.md) | KG reproduced (table above); Patrick profiles and Zhao depletion are the multiphase gates (V3/V4) | KG **done**; Patrick/Zhao planned for the two-fluid solver |
| **Zhao et al. 2008**, Ann Biomed Eng 36:1130–1141, doi:10.1007/s10439-008-9494-z, PMID 18418710, PMCID PMC6390961 (`verified-primary` abstract) | Planar 100→200 µm sudden expansion, Hct 20 %, Re 8.3–41.7; qualitative recirculation / RBC-depleted zone near the step | Qualitative CFL comparison for M2b. A 2D **single-phase** model cannot represent it (3D rectangular channel, depletion, depth): our planar E = 2 Newtonian run at Re 12.2 gives $x_r/h_1 = 0.536$ vs 3.07 — documented as CHECK, not comparable (`cfd/openfoam/validation_zhao2d.txt`) | qualitative only |
| **Sezer et al. 2026**, Micromachines 17(3):316, doi:10.3390/mi17030316, PMC13029181 (`verified`, full text read) | Euler–Euler (Fluent) in the Zhao expansion: $\Delta P^* = \Delta P/(\tfrac12 \rho_p u_i^2)$ correlation $\Delta P^* = 226.72\,\alpha_R^2 - 148.57\,\alpha_R + 117.37$ (Eq. 23, max rel. err 1.13 %, minimum at $\alpha_R \approx 0.3$), CFL definition Eq. 21 and CFL values (Fig. 13, figures only); Saffman–Mei lift, Gidaspow drag, Jung Carreau–Yasuda mixture viscosity (λ = 0.110 s) | Quantitative two-fluid target: ΔP*(α_R) trend + minimum location, CFL trend (2D vs their 3D quarter domain, factor `TODO`) | target for the in-progress v2 two-fluid work |
| **Gracka et al. 2022**, Comput Methods Programs Biomed 226:107117, doi:10.1016/j.cmpb.2022.107117 (CC BY; read in full → `verified`) | Hyperbolic contractions (throat 54 / 20 µm, Ht 5 %): experimental CFL ≈ **5 µm (A)** and **35 µm (B)** ~300 µm downstream of the throat; EE vs Euler–Lagrange comparison | CFL-thickness validation metric for the multiphase models (`validation/` targets 5/35 µm) | target for v2 |
| **Patrick et al.** (as used by Wu 2015 §3.2 and Sezer 2026 Fig. 7) | Measured RBC velocity profiles in a 100 µm-deep channel at Hct 0.45, z = 8 / 16 µm | RBC-phase velocity profiles gate for M2b (data live in the Sezer figures) | need the data extracted |
| **Fedosov et al. 2010**, Microcirculation 17:615, doi:10.1111/j.1549-8719.2010.00056.x, PMC3529161 (`verified`, full text) | DPD deformable-RBC tubes D = 10–40 µm, Ht 0.15–0.45: CFL wider at lower Ht and larger D (D = 10 µm, Ht 0.45: 1.19–1.45 µm); weak shear-rate dependence; in-vivo CFL 0.5–4 µm; Pries Fåhræus relation Eq. 12 | Trends + calibration data for the V4 wall-lift closure ([§2-theory](2-theory.md)); CFL(D, Ht) curves in Fig. 10 (digitise) | used as closure evidence |

Mesh-independence gate (all families): Roache 3-level GCI, $r = 1.5$, $F_s = 1.25$, thresholds < 2 % on
$\Delta p^*$ and $x_r/h_1$, < 5 % on $f_{window}$ — 30/30 runs converged, full table in
`cfd/openfoam/gci_summary.md`, figure `figures/fig06_gci.png`. Production mesh: level 2 for all six families
(with five documented FLAGs, e.g. $x_r/h_1$ not 2 %-certified for extreme shear-thinning in G2/G3, and
`max τ*_w` cell-size-dependent at sharp corners — **not** a corpus target).

Cross-method target: L1 (finite-volume) vs L2 (3D LBM in 2D mode) must agree within 5 % relative L2 on
u, τ, φ ([§4-validation](4-validation.md), threshold `TODO: confirm`).

---

## Validation targets (must pass before data generation)


| Case | Quantity | Target | Source |
|---|---|---|---|
| Sudden expansion | Reattachment length | 224 µm (2023 model), 230 µm (literature) | 2023 deck slide 7 |
| Straight channel | Radius of max encapsulation | 0.56R (2023), ≈0.6R (Piergiovanni 2020) | slides 4, 8 |
| Benchmark micro-channels | Hct profile, CFL thickness | Wu et al. 2015 figures | 02-literature |
| Hyperbolic contraction | CFL thickness | 5 µm (broad) / 35 µm (narrow) | 2022 CFL study |
| Any | Mesh independence | < 2 % change in max τ and mean I at next refinement | standard practice |
| L1 vs L2 (2D mode) | u, τ, φ fields | relative L2 difference < 5 % `TODO: confirm threshold` | cross-method |
If the 2023 COMSOL results are found, add their fields as extra targets.

---

## Constants verification (task T0, 2026-11-24, the team)


Rule (from [§2-theory](2-theory.md)): every constant has a source from `knowledge/` or is marked
`TODO: verify` / `TODO: source`. Statuses used below: `verified-primary` (original source read in full —
only possible for open-access full texts), `secondary` (value as quoted by a named citing paper; original not
read), `not found` (could not be located/read — nothing invented), `TODO: source`.

Web sources used 2026-11-24: NCBI BioC full-text API (PMC4521229 = Wu 2015 author manuscript, read in full),
EuropePMC metadata/abstract API (PMID 18418710 = Zhao 2008 abstract), AIP/OSTI records, and citing papers
listed per row. No original PDFs were added to `knowledge/papers/pdf/`.

### Table

| # | Constant | Value in repo | Value in source | Citation (author, year, journal, table/eq/page) | Status |
|---|---|---|---|---|---|
| 1 | Carreau μ₀ | 0.056 Pa·s | 0.056 Pa·s | Cho & Kensey 1991, *Biorheology* 28(3–4):241–262, Table 1 (as quoted) | `secondary` — arXiv:2502.21168; Kumar et al. 2018 (Comp. Biol. Med. 92:168–177) |
| 2 | Carreau μ∞ | 0.00345 Pa·s | 0.00345 Pa·s (one citing paper rounds to 0.0035) | same as #1 | `secondary` — same citers |
| 3 | Carreau λ | 3.313 s | 3.313 s | same as #1 | `secondary` — same citers |
| 4 | Carreau n | 0.3568 | 0.3568 | same as #1 | `secondary` — same citers |
| 5 | Carreau–Yasuda a | 2 | a = 2 (Yasuda exponent; the 1991 paper uses the Carreau form, i.e. a = 2) | same as #1 | `secondary` — Kumar et al. 2018 quote "Yasuda exponent a = 2" |
| 6 | Hemolysis C, α, β (HI in %, τ in Pa, t in s) | C = 3.62e-5, α = 2.416, β = 0.785 | C = 3.62e-5, α = 2.416, β = 0.785; fit range τ = 120–535 Pa, t = 14–700 ms (Couette) | Giersiepen M, Wurzinger LJ, Opitz R, Reul H 1990, *Int J Artif Organs* 13:300–306 | `secondary` — PMC10130582, PMC13291019 (Springer 2026 review); C = 3.62e-7 if HI as fraction |
| 7 | Phillips migration flux form N = −K_c a²φ∇(γ̇φ) − K_μ a²φ²γ̇∇(ln μ) | same | same two-term flux ("diffusive flux model") | Phillips RJ, Armstrong RC, Brown RA, Graham AL, Abbott JR 1992, *Phys Fluids A* 4(1):30–40 | `secondary` — AIP abstract page + OSTI 5551805 confirm paper & flux; form quoted by many citers |
| 8 | Phillips K_c, K_μ | K_c ≈ 0.41, K_μ ≈ 0.62 | K_c = 0.41, K_μ = 0.62 as used by later authors; some quote 0.435 for pipe flow (geometry-dependent fits) | Phillips et al. 1992, *Phys Fluids A* 4:30–40, as quoted e.g. in "Numerical simulations of particle migration… frame-invariant formulation" (ResearchGate) | `secondary` — exact fitted values not re-read from original |
| 9 | Garon–Farinas Eulerian damage D_I = HI^(1/β), u·∇D_I = (C τ^α)^(1/β) | same | linearization of the time variable; Eulerian advection for HI^(1/β) | Garon A, Farinas M-I 2004, *Artificial Organs* 28(11):1016–1025, DOI 10.1111/j.1525-1594.2004.00026.x | `secondary` — form as quoted by citing papers (e.g. ResearchGate 8213352) |
| 10 | Zhao 2008 inlet width / expansion ratio | CORPUS_PLAN says h₁ = 75 µm | inlet 100 µm : outlet 200 µm, expansion ratio E = 2 | Zhao R, Marhefka JN, Shu F, Hund SJ, Kameneva MV, Antaki JF 2008, *Ann Biomed Eng* 36(7):1130–1141, DOI 10.1007/s10439-008-9494-z, PMID 18418710, PMCID PMC6390961, abstract | `verified-primary` (abstract read via EuropePMC; full text not OA). **Discrepancy:** repo's 75 µm comes from the owner's 2023 slides (slide 7), not from Zhao 2008 |
| 11 | Zhao 2008 Hct | CORPUS_PLAN says φ = 0.22 | Hct 20% studied (and ≥ 30%); "at all flow rates … and hematocrit 20% and lower" | same abstract | `verified-primary` (abstract). Repo's 0.22 again from 2023 slides |
| 12 | Zhao 2008 Re definition & range | CORPUS_PLAN says Re = 12.2 | abstract: "all flow rates studied (Re = 8.3–41.7)"; secondary: U_in = 0.167 m/s → Re = 30.5, U_in = 0.833 m/s → Re = 152; lengths 1 mm before / 2 mm after the step | Zhao 2008 abstract; Wu, Aubry, Antaki, Massoudi 2018, *J Comput Phys* 376:112856 (PMC8545272, read in full) | range `verified-primary` (abstract); Re definition `not found` in abstract; U/Re pairs `secondary` (Wu 2018) |
| 13 | Zhao 2008 reattachment length | 224 µm (2023 model) vs 230 µm ("literature") | no numeric reattachment length reported — visualization study; RBC-depleted zone "comparable with the size of the cells" inside the step | Zhao 2008 abstract | `not found` in Zhao 2008; the 224/230 µm pair is from the owner's 2023 slides slide 7 (ref [8] = Wu 2015) |
| 14 | Zhao 2008 channel depth | not in repo | not in abstract; Wu 2015 z-averages their simulation "to account for the finite depth-of-field" of Zhao's images | Zhao 2008 abstract; Wu 2015 §3.3 | `not found` |
| 15 | Wu et al. 2015 OpenFOAM benchmark | "IJES 95:49" | *Int J Eng Sci* 95:49–59, DOI 10.1016/j.ijengsci.2015.06.004, PMID 26240438, PMCID PMC4521229 (author manuscript NIHMS705281, **read in full** via NCBI BioC): two-fluid OpenFOAM solver; case 3.1 = Karino & Goldsmith 1977 tubular expansion 77.5→252 µm (E = 3.25), 1 mm/5 mm, water + 1% hardened RBCs, U = 0.0757 m/s (Re = 12.2) and 0.233 m/s (Re = 37.8); case 3.3 = Zhao et al. rectangular sudden expansion (Fig 7), Hct = 0.20, U_in = 0.167 & 0.833 m/s, 125 544 nodes, z-averaged; qualitative agreement on recirculation-zone size/intensity | Wu W-T, Yang F, Antaki JF, Aubry N, Massoudi M 2015, *Int J Eng Sci* 95:49–59 | `verified-primary` (manuscript full text read). **Note:** Re = 12.2 in Wu 2015 belongs to the Karino & Goldsmith *tubular* case, not the Zhao case |
| 16 | Physiological range β = μ₀/μ∞ across patients (corpus design) | 5–30 (`TODO: source`) | not found | — | `TODO: source` (single Cho & Kensey fit gives β = 0.056/0.00345 ≈ 16.2) |
| 17 | Physiological range n across patients (corpus design) | 0.25–0.60 (`TODO: source`) | not found | — | `TODO: source` (single fit n = 0.3568) |

### Correction
Row 15: Wu 2015 Fig. 2 gives tube **radii** R₁ = 0.0755 mm, R₂ = 0.252 mm (d₁ = 151 µm, D₂ = 504 µm, E = 3.3377),
upstream 1 mm / downstream 2 mm. Cross-check: Re = ρUd₁/μ = 1027·0.0757·151e-6/0.96e-3 = 12.23 ✔ (Re is d₁-based).
Targets (x_r from the step along the outer wall, Wu 2015 Fig. 3): 230 µm at Re 12.2 → x_r/d₁ = 1.523;
715 µm at Re 37.8 → x_r/d₁ = 4.735. The 2023 slide's "inlet 0.075 mm" is the radius R₁.

### Discrepancy flags (for (internal planning notes) / the team)

1. [3-methods](3-methods.md) attributes "h₁ = 75 µm, φ = 0.22, Re = 12.2, reattachment 224–230 µm" to Zhao 2008.
   The primary source (Zhao 2008 abstract) says 100:200 µm, Hct 20%, Re = 8.3–41.7. The 75 µm / 0.22 / 12.2 /
   224–230 µm values come from the owner's 2023 deck (slide 7: "Inlet: 0.075mm, Ht: 0.22, Re: 12.2", "Flow
   Reattachment Point: 224 um (230 um - Literature)"), whose ref [8] is Wu et al. 2015. In Wu 2015, Re = 12.2
   is the Karino & Goldsmith 77.5 µm *tube* case. The 230 µm "literature" value could not be traced to a
   numeric source (`not found`).
2. Zhao 2008 is a 3D rectangular channel at Hct 20–30% with RBC depletion (cell-free layer) — a 2D
   single-phase model cannot represent the Hct/depth physics; only the recirculation length is comparable.

### Method notes

- Wu 2015 full text: `https://www.ncbi.nlm.nih.gov/research/bionlp/RESTful/pmcoa.cgi/BioC_json/PMC4521229/unicode`
- Zhao 2008 abstract: EuropePMC `EXT_ID:18418710` (PMID 18418710, PMCID PMC6390961, not in OA full-text subset).
- Cho & Kensey 1991 and Giersiepen 1990 originals are paywalled and not in `knowledge/papers/pdf/` → values
  above are `secondary`, quoted by the named citing papers.

---

## Assumption register (math ↔ biology/experiment) and validation cases


Owner rule (2026-11-24): **every mathematical assumption is linked to a biological explanation or an experimental
finding**, with its source and status. Labels: `verified` = original full text read (section/Eq./Fig.);
`secondary` = stated in a paper we read but citing another source; `abstract` = abstract only; `TODO` = no source yet.

### Sources read 2026-11-24
| Key | Citation | Access | Use |
|---|---|---|---|
| Wu15 | Wu et al., Int J Eng Sci 95:49 (2015), PMC4521229 | full | closures, KG / Zhao benchmarks ([§2-theory](2-theory.md)) |
| Sezer26 | Sezer, Kaya, Tabatabaei Malazi, Dalkılıç, Micromachines 17(3):316 (2026), doi 10.3390/mi17030316, PMC13029181 | full | Euler–Euler closures, ΔP*(α_R), CFL in Zhao expansion |
| Bern25 | Bernardelli, Piergiovanni, Bianchi, Carlo-Stella, Costantino, Casagrande, Ann Biomed Eng (2025), doi 10.1007/s10439-025-03678-2, PMC12006216 | full | **microfluidic shear encapsulation into RBCs** (our application) |
| Amar26 | Amar et al., Small (2026), doi 10.1002/smll.74818, PMC13580257 | full | shear-triggered release, pore/hemolysis thresholds, areal-strain criterion |
| Gracka22 | Gracka, Lima, Miranda, Student, Melka, Ostrowski, Comput Methods Programs Biomed (2022), doi 10.1016/j.cmpb.2022.107117 | abstract (paywalled) | CFL validation in hyperbolic contractions, Euler–Euler vs Euler–Lagrange |
| Pier20 | Piergiovanni, Casagrande, Taverna et al., Ann Biomed Eng 48:236 (2020), doi 10.1007/s10439-019-02342-w | full (owner PDF) | encapsulation index Eq. 1, 50 µm channel, Ht 1–10 %, exposure 0.15–1.5 s, Tillman diagram |
| Zeibi23 | Zeibi Shirejini, Carberry, Alt, Gregory, Hagemeyer, Adv Funct Mater (2023), doi 10.1002/adfm.202303717 | metadata only | review, shear-responsive delivery (background) |
| Park25 | Park et al., Nat Commun (2025), doi 10.1038/s41467-025-65185-1, PMC12638801 | abstract | RBC hitchhiking in vivo — background only, not a flow case |
| COMSOL | COMSOL 6.3 CFD Module, Euler–Euler theory (09.158/09.159) | full | equation structure ([§2-theory](2-theory.md)) |

### Assumption register
| # | Assumption (where used) | Biological / experimental basis | Source, status | Strength |
|---|---|---|---|---|
| A1 | RBCs as a continuum phase (Euler–Euler V3/V4) | Two-fluid models reproduce measured RBC velocity profiles (Patrick et al.) and measured CFL (Gracka: 5 / 35 µm, CFL φ < 2 % vs 7–8 % core) in 100 µm-scale channels | Sezer26 §3.1 Fig. 7–8 verified; Gracka22 abstract | medium; needs d/h ≪ 1 → we use h ≥ 80 µm (d/h ≤ 0.1) |
| A2 | RBC = rigid sphere, d = 8 µm | Healthy RBC diameter ≈ 8 µm; 4 / 11 µm represent microcytic / megaloblastic pathologies; rigid-sphere is a stated simplification, deformability affects CFL | Sezer26 §1, §2.4, §3.5 verified (citing [25–27]) | weak for deformation; V4 wall-lift is the proxy |
| A3 | Plasma Newtonian, μ_p = 0.96 mPa·s, ρ 1027 / 1093 kg/m³ | Plasma has constant viscosity; values used by Wu15 and Sezer26 (via Kim et al.) | Wu15 §3, Sezer26 §2.2 verified (Sezer cites [8]) | medium; device buffers differ (A14) |
| A4 | Whole-blood shear-thinning comes from the RBC phase; mixture viscosity rises with φ | Experimental blood viscosity vs Ht fits: Brooks data (Wu Eq. 9), Jung & Hassanein Carreau–Yasuda fit (Sezer Eqs. 5–11, λ = 0.110 s) | Wu15 Eq. 9 verified (fit to Brooks); Sezer26 Eqs. 5–11 verified (citing [29,30]) | strong (data fits) |
| A5 | Stokes drag with hindrance f(φ) = e^{2.68φ} + φ^{0.43} | Particle Re ≪ 1 in microchannels; hindrance function is from Rusche & Issa (bubbly/particle flows), not RBC data | Wu15 Eq. 10 verified; RBC-specific data `TODO` | weak (closure borrowed) |
| A6 | Saffman shear lift drives cells from walls | CFL / Fåhræus–Lindqvist effect observed; mechanism disputed: shear-gradient, granular temperature, or deformability (Secomb & Pries) | Sezer26 §1 verified (lists the competing mechanisms, citing [2–9]) | weak mechanism, strong phenomenon → calibrate on CFL data |
| A7 | Shear window 30–60 Pa for encapsulation (2023 report) | Bern25: loading achieved at max wall shear 49.7 Pa (50 µL/min, 1 % Ht), sub-hemolytic. Amar26: transient pores ≈ 15 Pa, hemolysis threshold ≈ 45 Pa | Bern25 Discussion verified; Amar26 Intro secondary (citing [18,19]) | **FLAG: lower bound may be ~15 Pa; upper bound depends on exposure time — window and HI must be time-aware** |
| A8 | Encapsulation index I ∝ v·τ·φ (2023 Eq. 1) | Bern25: loaded fraction ~flat for 30→50 µL/min, attributed to RBCs in the core shielded by the plasma layer → local φ and CFL matter (supports Euler–Euler over uniform φ). Amar26: release correlates with membrane areal strain; pores above ~4 % areal strain (Evans 1976) | Bern25 Discussion verified; Amar26 Fig. 4E verified, Evans secondary | medium; strain-based index = future M3b |
| A9 | Hemolysis is dose × time (Giersiepen power law) | Bern25 sub-hemolytic at 49.7 Pa for ~0.18 s transit (L = 58.5 mm, U ≈ 0.33 m/s), above Amar's static 45 Pa → time dependence matters | Bern25 verified (transit time computed by us); Giersiepen constants secondary ([§2-theory](2-theory.md)) | medium |
| A10 | Steady laminar flow | Devices are syringe-pump driven at constant Q (Bern25 §Methods); steady models capture mean RBC behaviour (Recktenwald, via Sezer) | Bern25 verified; Sezer26 §1 secondary | strong |
| A11 | Developed inlet profile (owner choice) | Sezer26: entrance length 172.67 µm < 20 % of the inlet section → developed; uniform-φ inlet in their model | Sezer26 §2.4 Eq. 20 verified | medium; real chips have tubing → chip contractions |
| A12 | 2D planar (axisymmetric for KG) | Real channels are rectangular / square (Bern25 50×50 µm, Sezer26 100×100 µm) | verified | weak for absolute numbers → 2D↔3D factor `TODO` |
| A13 | Haematocrit range φ ∈ [0.2, 0.6] (corpus v2.1) | **Encapsulation devices run diluted suspensions: Bern25 1 % and 10 % Ht** (10 % at 5 µL/min clogged by sedimentation); Gracka22 5 % | Bern25 Table 2 verified; Gracka22 abstract | **FLAG: corpus must cover φ 0.01–0.1 (device regime) as well as 0.2–0.6 (whole blood)** |
| A14 | Carrier fluid = plasma | Bern25 uses PBS / GASP (PBS + 1 % BSA) + dextran; μ_susp 1.12–1.68 mPa·s (Table 3) | Bern25 Table 3 verified | → μ_carrier is a run parameter, not fixed to plasma |

### Validation cases (status)
| Case | Data | Our status |
|---|---|---|
| V-KG Karino & Goldsmith tube (via Wu15) | x_r/d₁ 1.523 / 4.735 | OpenFOAM M1 1.0 / 1.5 %; **FEniCSx two-fluid +2.9 / −1.3 %** (plasma viscosity) |
| V-Zhao/Sezer expansion | ΔP*(α_R) Eq. 23, ΔP* = ΔP/(½ρ_p u_i²) Eq. 22; δ*_CFL(z) Fig. 13 (figure only) | planned (trend + minimum location; 2D vs 3D quarter) |
| V-Patrick velocity profiles (used by Wu15, Sezer26) | RBC velocity at 8 / 16 µm from wall, Hct 0.45 | need the data (figures in Sezer Fig. 7) |
| V-Gracka hyperbolic contraction | CFL 5 / 35 µm at Ht 5 % | need full text (owner PDF) for geometry |
| V-Bernardelli encapsulation channel | 50 µm square × 58.5 mm, 30 / 50 µL/min, Ht 1 / 10 %, 49.7 Pa max, loaded 70–94 % | **functional check**: predicted window exposure and its flow-rate sensitivity vs the flat loaded fraction |
| V-Amar thresholds | pores ~15 Pa, hemolysis ~45 Pa, 4 % areal strain | thresholds for window/HI; areal strain needs M3b |

### Window decision (owner + the team + the team + the team, 2026-11-24; logs data/agent_logs/window_*.md)
Stress window = **screening statistic**; the decision criterion is **time-aware**. All metrics are post-processing
on stored fields (no reruns), weighted by the RBC flux φ·w along RBC paths (V3/V4) or u·φ_in (V1/V2, upper-bound bias),
never by wall stress. Star units: τ* = τ h₁/(μ_c U), t* = t U/h₁.
| ID | Metric | Why |
|---|---|---|
| W1 | f_window, 30–60 Pa | 2023 comparability |
| W2 | f_window, 15–45 Pa | Amar26 thresholds (secondary) |
| W3 | f_window, 1–7 Pa | physiological WSS — negative control, expect no loading |
| D15, D30 | one-sided gate τ ≥ τ_lo (15, 30 Pa): exposure t*_pore, loaded fraction f_load, excess dose J* = ∫max(0, τ* − τ*_lo) dt* | pore opening needs time above threshold, no static upper cap |
| HI | Giersiepen via Garon–Farinas: D*_I = ∫(τ*)^{α/β} dt*, HI = C (μ_c U/h₁)^α (h₁/U)^β (D*_I)^β; report mean and P95/P99 | hemolysis is dose × time (A9) |
| W4 | per-design effective upper bound τ_hi,eff from HI(τ, t_exp) = HI_lim | derives the cap instead of assuming it |
| rank | Spearman rank correlation of designs across W1/W2/D15/D30 | if rankings agree, the window choice is not decisive |
Checks: Bern25 49.7 Pa × 0.18 s → HI ≈ 0.12 %; 45 Pa reaches 0.5 % only after ≈ 1.5 s (t ∝ τ^−3.08) → Amar's 45 Pa is
a ~1 s number. At fixed geometry HI ∝ U^(α−β) = U^1.63 (Newtonian) — a falsifiable corpus check.
Open: HI_lim value (`TODO: source`); Giersiepen fitted on longer exposures than 0.18 s (extrapolation flagged);
Giersiepen constants and Evans 4 % areal strain still `secondary`; areal-strain criterion needs M3b.

### Legal/attribution note (2026-11-24)
COMSOL documentation is used for equations only (mathematical methods are not copyrightable); all equations are
written in our own notation, the original papers COMSOL cites are the primary references, COMSOL docs are cited as
software documentation; no COMSOL text, figures or equation images are copied into the repo or publications; no
endorsement implied. 2023 COMSOL results/report: check group/university rules before publishing them.

### A6 update (2026-11-24): what sustains the CFL
Evidence: (i) steady FEniCSx V4 (drag + Saffman lift, COMSOL pseudo-time recipe) converges with only 1 % wall
depletion; (ii) OpenFOAM EE (WenYu + LegendreMagnaudet) gives monotonic ΔP*(α), no Sezer minimum; (iii) owner's 2023
COMSOL time-dependent runs: CFL visible early, vanished later ([§6-comsol-2023](6-comsol-2023.md)). Physical chips keep a CFL (Gracka 2022:
5/35 µm; Bern25 discussion). ⇒ the CFL in the steady state requires **deformability-induced wall lift** (force ∝ shear
rate, independent of slip), not slip-based Saffman lift. V4 adds it, with a sourced form and a coefficient calibrated on
measured CFL thickness (`TODO: source` form and data before production).

### Piergiovanni et al. 2020 read in full (owner PDF, 2026-11-24) — primary source of the encapsulation index
Piergiovanni, Casagrande, Taverna et al., Ann Biomed Eng 48:236–246 (2020), doi 10.1007/s10439-019-02342-w — `verified`.
- CFD: **COMSOL 4.2a "Mixture Model, Laminar Flow"** (not Euler–Euler), rigid spheres d = 8 µm in PBS, gravity included;
  straight 50 µm square channel, mesh 6.9 M elements; Q 0.25–100 µL/min, **Ht 1–10 %** (dilute limit of the mixture
  model); Re 0.1–20; model predicts RBC deposition at 0.25 µL/min that disappears above 10 µL/min.
- **Eq. 1: I_encap = v·τ·Φ / (v_max·τ_max·Φ_max)**, maxima in a reference cross-section. At **0.6R: v = 0.5 m/s,
  τ = 90 Pa, uniform Φ → I = 0.31**; the crown where I > 0.2 covers **57.8 % of the cross-section**; τ_eff = mean τ on
  cells in that crown. The crown is attributed to the tubular-pinch (Segré–Silberberg) equilibrium at 0.6R for Re_p > 1.
- Validation: µ-PIV velocity profiles (Ht 1 %, 30 µL/min, centreline 0.3 m/s). Experiments: channel lengths 10 / 58.5 /
  87 / 117 mm, 5–50 µL/min, FITC-dextran; best encapsulation with the 58.5 mm channel = **exposure 0.15–1.5 s**; the
  117 mm channel produced haemolysis (fragments > 0.1 %). Haemolysis judged on a **Tillman force–time diagram**; "none
  of these models indicates a sharp threshold".
- **The "30–60 Pa" window is NOT stated in the text** (no such numbers in the paper body); it may come from Fig. 5 or
  elsewhere → A7 baseline 30–60 Pa stays `TODO: source` (owner to confirm where it came from). Supports A9 (dose × time).
**A7 source resolved (owner, 2026-11-24):** the 30–60 Pa window came from the owner's own 2023 COMSOL simulations
(Nalla & Gereige 2023), not from a paper → label `own 2023 simulation result`. Consistent with: Bernardelli 2025 loading at
peak wall shear 49.7 Pa; Piergiovanni 2020 example point 90 Pa at 0.6R (I = 0.31). Owner recollection that Piergiovanni
used Euler–Lagrange conflicts with the paper text ("Mixture Model, Laminar Flow", Methods) — check Supplementary S2.

---

## GCI study — 2D M1 mesh independence (T3, the team, 2026-11-24)


Roache three-level GCI, `r = 1.5`, safety factor `F_s = 1.25`, computed by `campaign/gci.py` from
`campaign/gci_design.jsonl` (10 cases = 2 extreme corners × families G2–G6, × 3 mesh levels = 30 runs).
All 30 runs converged (`converged_by: residual`), all `checkMesh` `Mesh OK.`.

- VM `rbc-team-3` (n2-highcpu-8 SPOT, europe-west4-a), OpenFOAM ESI v2306, 8 parallel workers.
- Commands:
  - `python3 -m campaign.run_batch --design campaign/gci_design.jsonl --tier gci --manifest-name manifest_gci.jsonl --workers 8 --work-dir ~/work/gci`
  - `python3 -m campaign.gci --manifest ~/work/gci/manifest_gci.jsonl --out-dir ~/work/gciout --upload-bucket gs://<corpus-bucket> --prefix corpus/v1`
- Artefacts: `gs://<corpus-bucket>/corpus/v1/gci/<family>/<case>.json` (+ `gci/summary.md`),
  fields in `gs://<corpus-bucket>/corpus/v1/runs/gci/<family>/<run_id>.npz`,
  manifest `gs://<corpus-bucket>/corpus/v1/manifest_gci.jsonl`.
- Corners: `inertial` = Newtonian, Re 25, worst geometry; `shearthin` = BirdCarreau β 30, n 0.25, Cu 2.0, a 2, Re 10.
- Thresholds ([3-methods](3-methods.md) §0, this task step 4): **< 2 % for Δp\* and x_r/h₁, < 5 % for f_window.**
  `GCI L0→L1` is the estimate for level 0 (level 0 vs the next finer), `GCI L1→L2` the estimate for level 1.

### Full table

| Case | Family | n_cells L0/L1/L2 | Scalar | L0 | L1 | L2 | p | GCI L0→L1 (%) | GCI L1→L2 (%) | Conv |
|---|---|---|---|---|---|---|---|---|---|---|
| GCI_G2_inertial | G2 | 16000/36000/81000 | Δp* | 59.46 | 59.42 | 59.41 | 4.28 | 0.02 | 0.00 | monotonic |
| GCI_G2_inertial | G2 | 16000/36000/81000 | x_r/h₁ | 4.479 | 4.423 | 4.441 | 2.84 | 0.72 | 0.23 | oscillatory |
| GCI_G2_inertial | G2 | 16000/36000/81000 | max τ*_w | 6.62 | 7.166 | 7.835 | - | - | - | divergent |
| GCI_G2_inertial | G2 | 16000/36000/81000 | f_window | 0 | 0 | 0 | - | 0.00 | 0.00 | exact |
| GCI_G2_shearthin | G2 | 16000/36000/81000 | Δp* | 662.8 | 665.6 | 667.1 | 1.52 | 0.62 | 0.33 | monotonic |
| GCI_G2_shearthin | G2 | 16000/36000/81000 | x_r/h₁ | 0.5869 | 0.5832 | 0.5863 | 0.47 | 3.87 | 3.18 | oscillatory |
| GCI_G2_shearthin | G2 | 16000/36000/81000 | max τ*_w | 60.49 | 74.85 | 94.59 | - | - | - | divergent |
| GCI_G2_shearthin | G2 | 16000/36000/81000 | f_window | 0.2206 | 0.2205 | 0.2464 | - | - | - | divergent |
| GCI_G3_inertial | G3 | 2976/6696/15066 | Δp* | 767.6 | 773.1 | 776.2 | 1.50 | 1.08 | 0.58 | monotonic |
| GCI_G3_inertial | G3 | 2976/6696/15066 | x_r/h₁ | 0.9571 | 0.9301 | 0.9303 | 12.28 | 0.03 | 0.00 | oscillatory |
| GCI_G3_inertial | G3 | 2976/6696/15066 | max τ*_w | 90.27 | 101.9 | 115.6 | - | - | - | divergent |
| GCI_G3_inertial | G3 | 2976/6696/15066 | f_window | 0.3516 | 0.3515 | 0.3515 | 1.33 | 0.01 | 0.01 | oscillatory |
| GCI_G3_shearthin | G3 | 2976/6696/15066 | Δp* | 2320 | 2330 | 2337 | 0.88 | 1.30 | 0.91 | monotonic |
| GCI_G3_shearthin | G3 | 2976/6696/15066 | x_r/h₁ | 0.09135 | 0.07738 | 0.08124 | 3.17 | 8.62 | 2.27 | oscillatory |
| GCI_G3_shearthin | G3 | 2976/6696/15066 | max τ*_w | 142.8 | 154.3 | 169.7 | - | - | - | divergent |
| GCI_G3_shearthin | G3 | 2976/6696/15066 | f_window | 0.06308 | 0.06291 | 0.06282 | 1.47 | 0.41 | 0.23 | monotonic |
| GCI_G4_inertial | G4 | 1536/3456/7776 | Δp* | 328.4 | 332.4 | 334 | 2.26 | 1.01 | 0.40 | monotonic |
| GCI_G4_inertial | G4 | 1536/3456/7776 | max τ*_w | 24.87 | 30.93 | 37.72 | - | - | - | divergent |
| GCI_G4_inertial | G4 | 1536/3456/7776 | f_window | 0.2357 | 0.2346 | 0.2343 | 2.55 | 0.31 | 0.11 | monotonic |
| GCI_G4_shearthin | G4 | 1536/3456/7776 | Δp* | 1606 | 1624 | 1629 | 2.96 | 0.59 | 0.18 | monotonic |
| GCI_G4_shearthin | G4 | 1536/3456/7776 | max τ*_w | 71.4 | 85.39 | 101.7 | - | - | - | divergent |
| GCI_G4_shearthin | G4 | 1536/3456/7776 | f_window | 0 | 0 | 0.09162 | - | - | - | divergent |
| GCI_G5_inertial | G5 | 1968/4428/9963 | Δp* | 313.4 | 314.9 | 315.4 | 3.18 | 0.24 | 0.07 | monotonic |
| GCI_G5_inertial | G5 | 1968/4428/9963 | max τ*_w | 33.26 | 41.01 | 49.57 | - | - | - | divergent |
| GCI_G5_inertial | G5 | 1968/4428/9963 | f_window | 0.2336 | 0.2332 | 0.233 | 2.64 | 0.12 | 0.04 | monotonic |
| GCI_G5_shearthin | G5 | 1968/4428/9963 | Δp* | 1582 | 1589 | 1591 | 3.67 | 0.17 | 0.04 | monotonic |
| GCI_G5_shearthin | G5 | 1968/4428/9963 | max τ*_w | 81.62 | 97.39 | 115.9 | - | - | - | divergent |
| GCI_G5_shearthin | G5 | 1968/4428/9963 | f_window | 0 | 0.0751 | 0.06744 | 5.63 | 14.19 | 1.61 | oscillatory |
| GCI_G6_inertial | G6 | 4928/11088/24948 | Δp* | 483.2 | 484.4 | 484.9 | 2.21 | 0.21 | 0.09 | monotonic |
| GCI_G6_inertial | G6 | 4928/11088/24948 | max τ*_w | 9.731 | 9.854 | 9.916 | 1.71 | 1.56 | 0.78 | monotonic |
| GCI_G6_inertial | G6 | 4928/11088/24948 | f_window | 0.3517 | 0.3516 | 0.3515 | - | - | - | divergent |
| GCI_G6_shearthin | G6 | 4928/11088/24948 | Δp* | 3163 | 3174 | 3175 | 6.00 | 0.04 | 0.00 | monotonic |
| GCI_G6_shearthin | G6 | 4928/11088/24948 | max τ*_w | 60.49 | 74.85 | 94.59 | - | - | - | divergent |
| GCI_G6_shearthin | G6 | 4928/11088/24948 | f_window | 0.09516 | 0.06307 | 0.06284 | 12.12 | 0.47 | 0.00 | monotonic |

`max τ*_w` is **not** one of the T3 criteria (Δp\*, x_r/h₁, f_window only). It is reported because it is
monotonically rising with mesh refinement in 9 of 10 corners (only G6 inertial is monotonic-convergent): the
wall-shear maximum is set by the finest scale captured and keeps rising as the mesh refines (e.g. G5 shearthin
81.6 → 97.4 → 115.9). Physically that is the expected `τ_w = μ γ̇` peak on a sharp corner; it is a
*cell-size-dependent* quantity and must not be used as a corpus target without a surface-resolved metric.
Flagged for the team.

### Production level per family

Rule (task step 4): the **lowest** level whose GCI (that level vs the next finer) is < 2 % for Δp\* and
x_r/h₁ and < 5 % for f_window, evaluated on **both** corners of the family; if none qualifies, level 2 + FLAG.

| Family | L0 fails on | L1 fails on | Lowest qualifying | **Chosen** | Note |
|---|---|---|---|---|---|
| G2 | shearthin x_r 3.87 %, shearthin f_window (divergent) | shearthin x_r 3.18 %, shearthin f_window (divergent) | none | **2** | FLAG 1 |
| G3 | shearthin x_r 8.62 % | shearthin x_r 2.27 % | none | **2** | FLAG 1 |
| G4 | shearthin f_window (divergent: 0 / 0 / 0.0916) | shearthin f_window (divergent) | none | **2** | FLAG 2 |
| G5 | shearthin f_window 14.19 % | *literal* pass (1.61 %) | L1 by the letter | **2** | FLAG 3 |
| G6 | inertial f_window (divergent) | inertial f_window (divergent) | none | **2** | FLAG 4 |
| G7 | — | — | no GCI case designed | **2** | FLAG 5 |

All 384 M1 runs were therefore executed at **level 2** (uniform level ⇒ comparable resolution across families),
`campaign/m1_design.jsonl`, tier `m1`, one `--level 2 --ids <family ids>` call per family.

### FLAGs raised by this study

1. **G2, G3 — x_r/h₁ does not reach 2 % even at L1→L2** (G2 shearthin 3.18 %, G3 shearthin 2.27 %) in the
   strongest-shear-thinning corner. Δp\* (0.33 %, 0.91 %) and f_window are fine there. Reattachment length in
   these geometries is a bubble-edge quantity resolved by a handful of cells; a 4th level (L3, 2.25× again:
   G2 182 k cells) or a wall-normal refinement near the step corner would be needed to certify it. Recorded as
   `xr_over_h1` with the GCI level in the manifest — do not treat it as a 2 %-certified target for β→30, n→0.25.
2. **G4 — f_window is exactly 0 on L0 and L1 and 0.0916 on L2** in the shear-thinning corner: the 30–60 Pa
   window is not resolved at all on the two coarser meshes, so no GCI exists (divergence flag is genuine here).
   G4 has no `xr_over_h1` (bifurcation, no reattachment bubble).
3. **G5 — the ROACHE ESTIMATE IS MISLEADING, level 2 chosen against the literal rule.** f_window = 0 (L0) /
   0.0751 (L1) / 0.0674 (L2): the L0 value is degenerate (the window region is entirely unresolved), which
   inflates the observed order to p = 5.63 and drives GCI_fine down to 1.61 %. The *direct* L1→L2 change is
   **11.4 %**, i.e. three times the 5 % requirement, so level 1 is not mesh-independent for this scalar.
   Cost of the conservative choice: 4 428 → 9 963 cells/run for G5. the team/owner may relax G5 to level 1 if the
   literal GCI is preferred; the two G5 runs sets differ only by mesh density.
4. **G6 — f_window 0.3517 / 0.3516 / 0.3515**: the flag is `divergent` only because the successive differences
   grow (e32/e21 ≈ 0.67 ⇒ negative observed order) on a **0.04 %** total variation. Numerically this scalar is
   converged at L0; the level-2 choice for G6 is therefore conservative, not necessary. Kept at 2 for
   consistency with the rule.
5. **G7 has no GCI case** (`campaign/design_gci.py` covers G2–G6 only, per [3-methods](3-methods.md) §3). Level 2 is
   used as the documented fallback. A G7 corner pair is the missing piece if the corpus is ever re-meshed.

---

## Paired-design GCI study — 2D V1/V2 mesh independence at Re ≤ 60, φ ≤ 0.6, τ_w ≤ 150 Pa (T10 Part 3, the team, 2026-11-24)


Roache three-level GCI (`r = 1.5`, safety factor `F_s = 1.25`, `campaign/gci.py`) across mesh levels **L1, L2, L3**
on `campaign/gci_paired_design.jsonl` (6 families G2–G7 × 2 extremes `[max_Re, max_phi]` × 2 variants `[N, GN]` × 3 levels = **72 runs**, tier `gci_paired`).

- **VM:** `rbc-team-3` (`n2-custom-24-24576` SPOT, 24 vCPU, 24 GiB RAM, `europe-west4-a`), OpenFOAM ESI v2306, `--workers 24`.
- **Commands:**
  - `python3 -m campaign.design_gci_paired` → `campaign/gci_paired_design.jsonl` (72 runs)
  - `python3 -m campaign.run_batch --design campaign/gci_paired_design.jsonl --tier gci_paired --manifest-name manifest_gci_paired.jsonl --work-dir ~/work_gci_paired --workers 24 --timeout 3600`
  - Single retry (`relax_p = 0.2, relax_u = 0.5`) on the 2 initial L3 `not_converged` runs (`GCIP_G5_maxPhi_GN_L3` recovered `ok`; `GCIP_G5_maxRe_GN_L3` remained `not_converged`).
- **Outcomes:** **71/72 `ok`**, **1/72 `failed`** (`GCIP_G5_maxRe_GN_L3`, `not_converged`), **72/72 archived** (archive count = run count; 211.0 MiB archives, 83.8 MiB `.npz`), **23/24 complete L1/L2/L3 triples**.
- **Artefacts:** `gs://<corpus-bucket>/corpus/v1/manifest_gci_paired.jsonl`, `runs/gci_paired/<family>/`, `cases/gci_paired/<family>/`, `gci_paired/<family>/<case>.json`, `gci_paired/summary.md`.
- **Window-fraction definitions** (recomputed from each `.npz`'s stored `outlet_tau`, `outlet_u_n`, `outlet_weights` using the design's own `tau_ref = mu_carrier * U / h`, per [§4-validation](4-validation.md)):
  - `f_window_15_45` (W2: 15–45 Pa), `f_window_30_60` (W1: 30–60 Pa), `f_ge_15` (D15: τ ≥ 15 Pa), `f_ge_30` (D30: τ ≥ 30 Pa).
- **Rule (T10 step 8):** if a family has `GCI_fine > 5 %` on `Δp*` or `x_r/h₁`, list its V1/V2 `run_id`s as `needs_level3` for the team decision (no level-3 production re-runs executed here).

### Selected designs (from `campaign/paired_design.jsonl`)

| Family | Corner | Design ID | Re | φ | τ_w,nom (Pa) | τ_ref (Pa) | Cells L1 / L2 / L3 |
|---|---|---|---:|---:|---:|---:|---|
| G2 | `max_Re` | `D011` | 57.92 | 0.0754 | 98.0 | 13.862 | 14400/32400/72414 |
| G2 | `max_phi` | `B010` | 3.51 | 0.5948 | 12.0 | 0.248 | 6840/15390/35006 |
| G3 | `max_Re` | `D001` | 51.51 | 0.0178 | 50.0 | 8.032 | 5940/13365/30260 |
| G3 | `max_phi` | `B011` | 6.74 | 0.5839 | 20.7 | 0.481 | 6156/13851/31300 |
| G4 | `max_Re` | `D002` | 55.66 | 0.0853 | 78.7 | 10.883 | 3456/7776/17496 |
| G4 | `max_phi` | `B005` | 12.57 | 0.5647 | 28.3 | 0.748 | 3456/7776/17496 |
| G5 | `max_Re` | `B008` | 52.43 | 0.3837 | 66.9 | 3.982 | 3312/7452/16794 |
| G5 | `max_phi` | `B005` | 33.32 | 0.5929 | 66.2 | 1.537 | 3312/7452/16794 |
| G6 | `max_Re` | `D001` | 58.01 | 0.0112 | 37.0 | 6.018 | 8784/19764/44442 |
| G6 | `max_phi` | `B009` | 10.15 | 0.5967 | 48.6 | 1.091 | 22608/50868/114210 |
| G7 | `max_Re` | `B003` | 53.23 | 0.4657 | 53.9 | 2.335 | 13320/29970/66920 |
| G7 | `max_phi` | `B002` | 4.33 | 0.5775 | 15.3 | 0.362 | 19152/43092/97038 |

### Full Roache GCI table (L1 → L2 → L3, r = 1.5, F_s = 1.25)

`e32` = direct relative change L1→L2 (%), `e21` = direct relative change L2→L3 (%), `GCI_12` = `GCI_medium` (L1→L2, %), `GCI_23 (fine)` = `GCI_fine` (L2→L3, %).

| Case | Family | Var | n_cells L1/L2/L3 | Scalar | L1 | L2 | L3 | p | e32 (%) | e21 (%) | GCI_12 (%) | GCI_23 fine (%) | Conv |
|---|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| `GCIP_G2_maxRe_GN` | G2 | GN | 14400/32400/72414 | `dp_star` | 100.24 | 101.8 | 103.35 | 0.02 | 1.53 | 1.50 | 240.42 | 234.94 | monotonic |
| `GCIP_G2_maxRe_GN` | G2 | GN | 14400/32400/72414 | `xr_over_h1` | 2.8749 | 2.9059 | 2.9067 | 9.13 | 1.07 | 0.03 | 0.03 | 0.00 | monotonic |
| `GCIP_G2_maxRe_GN` | G2 | GN | 14400/32400/72414 | `f_window_15_45` | 0.015714 | 0.022259 | 0.02226 | 22.44 | 29.41 | 0.00 | 0.00 | 0.00 | monotonic |
| `GCIP_G2_maxRe_GN` | G2 | GN | 14400/32400/72414 | `f_window_30_60` | 0 | 0 | 0 | - | 0.00 | 0.00 | 0.00 | 0.00 | exact |
| `GCIP_G2_maxRe_GN` | G2 | GN | 14400/32400/72414 | `f_ge_15` | 0.015714 | 0.022259 | 0.02226 | 22.44 | 29.41 | 0.00 | 0.00 | 0.00 | monotonic |
| `GCIP_G2_maxRe_GN` | G2 | GN | 14400/32400/72414 | `f_ge_30` | 0 | 0 | 0 | - | 0.00 | 0.00 | 0.00 | 0.00 | exact |
| `GCIP_G2_maxRe_N` | G2 | N | 14400/32400/72414 | `dp_star` | 71.491 | 71.425 | 71.401 | 2.44 | 0.09 | 0.03 | 0.07 | 0.03 | monotonic |
| `GCIP_G2_maxRe_N` | G2 | N | 14400/32400/72414 | `xr_over_h1` | 2.882 | 2.9112 | 2.911 | 12.97 | 1.00 | 0.01 | 0.01 | 0.00 | oscillatory |
| `GCIP_G2_maxRe_N` | G2 | N | 14400/32400/72414 | `f_window_15_45` | 0.015706 | 0.015732 | 0.015729 | 5.48 | 0.16 | 0.02 | 0.02 | 0.00 | oscillatory |
| `GCIP_G2_maxRe_N` | G2 | N | 14400/32400/72414 | `f_window_30_60` | 0 | 0 | 0 | - | 0.00 | 0.00 | 0.00 | 0.00 | exact |
| `GCIP_G2_maxRe_N` | G2 | N | 14400/32400/72414 | `f_ge_15` | 0.015706 | 0.015732 | 0.015729 | 5.48 | 0.16 | 0.02 | 0.02 | 0.00 | oscillatory |
| `GCIP_G2_maxRe_N` | G2 | N | 14400/32400/72414 | `f_ge_30` | 0 | 0 | 0 | - | 0.00 | 0.00 | 0.00 | 0.00 | exact |
| `GCIP_G2_maxPhi_GN` | G2 | GN | 6840/15390/35006 | `dp_star` | 878.23 | 891.01 | 903.28 | 0.10 | 1.43 | 1.36 | 42.07 | 39.80 | monotonic |
| `GCIP_G2_maxPhi_GN` | G2 | GN | 6840/15390/35006 | `xr_over_h1` | 0.039158 | 0.13282 | 0.13316 | 13.88 | 70.52 | 0.25 | 0.32 | 0.00 | monotonic |
| `GCIP_G2_maxPhi_GN` | G2 | GN | 6840/15390/35006 | `f_window_15_45` | 0 | 0 | 0 | - | 0.00 | 0.00 | 0.00 | 0.00 | exact |
| `GCIP_G2_maxPhi_GN` | G2 | GN | 6840/15390/35006 | `f_window_30_60` | 0 | 0 | 0 | - | 0.00 | 0.00 | 0.00 | 0.00 | exact |
| `GCIP_G2_maxPhi_GN` | G2 | GN | 6840/15390/35006 | `f_ge_15` | 0 | 0 | 0 | - | 0.00 | 0.00 | 0.00 | 0.00 | exact |
| `GCIP_G2_maxPhi_GN` | G2 | GN | 6840/15390/35006 | `f_ge_30` | 0 | 0 | 0 | - | 0.00 | 0.00 | 0.00 | 0.00 | exact |
| `GCIP_G2_maxPhi_N` | G2 | N | 6840/15390/35006 | `dp_star` | 847.57 | 848.33 | 849.02 | 0.23 | 0.09 | 0.08 | 1.14 | 1.04 | monotonic |
| `GCIP_G2_maxPhi_N` | G2 | N | 6840/15390/35006 | `xr_over_h1` | 0.071255 | 0.16107 | 0.16139 | 13.85 | 55.76 | 0.20 | 0.25 | 0.00 | monotonic |
| `GCIP_G2_maxPhi_N` | G2 | N | 6840/15390/35006 | `f_window_15_45` | 0 | 0 | 0 | - | 0.00 | 0.00 | 0.00 | 0.00 | exact |
| `GCIP_G2_maxPhi_N` | G2 | N | 6840/15390/35006 | `f_window_30_60` | 0 | 0 | 0 | - | 0.00 | 0.00 | 0.00 | 0.00 | exact |
| `GCIP_G2_maxPhi_N` | G2 | N | 6840/15390/35006 | `f_ge_15` | 0 | 0 | 0 | - | 0.00 | 0.00 | 0.00 | 0.00 | exact |
| `GCIP_G2_maxPhi_N` | G2 | N | 6840/15390/35006 | `f_ge_30` | 0 | 0 | 0 | - | 0.00 | 0.00 | 0.00 | 0.00 | exact |
| `GCIP_G3_maxRe_GN` | G3 | GN | 5940/13365/30260 | `dp_star` | 284.92 | 286.9 | 288.68 | 0.26 | 0.69 | 0.62 | 7.75 | 6.93 | monotonic |
| `GCIP_G3_maxRe_GN` | G3 | GN | 5940/13365/30260 | `xr_over_h1` | 0.25504 | 0.26084 | 0.26731 | - | 2.22 | 2.42 | - | - | divergent |
| `GCIP_G3_maxRe_GN` | G3 | GN | 5940/13365/30260 | `f_window_15_45` | 0.56199 | 0.56184 | 0.56188 | 3.28 | 0.03 | 0.01 | 0.01 | 0.00 | oscillatory |
| `GCIP_G3_maxRe_GN` | G3 | GN | 5940/13365/30260 | `f_window_30_60` | 0.19925 | 0.19898 | 0.19896 | 7.38 | 0.14 | 0.01 | 0.01 | 0.00 | monotonic |
| `GCIP_G3_maxRe_GN` | G3 | GN | 5940/13365/30260 | `f_ge_15` | 0.57412 | 0.574 | 0.57406 | 1.72 | 0.02 | 0.01 | 0.03 | 0.01 | oscillatory |
| `GCIP_G3_maxRe_GN` | G3 | GN | 5940/13365/30260 | `f_ge_30` | 0.19925 | 0.19898 | 0.19896 | 7.38 | 0.14 | 0.01 | 0.01 | 0.00 | monotonic |
| `GCIP_G3_maxRe_N` | G3 | N | 5940/13365/30260 | `dp_star` | 258.17 | 258.8 | 259.16 | 1.32 | 0.24 | 0.14 | 0.43 | 0.25 | monotonic |
| `GCIP_G3_maxRe_N` | G3 | N | 5940/13365/30260 | `xr_over_h1` | 0.25522 | 0.26091 | 0.26735 | - | 2.18 | 2.41 | - | - | divergent |
| `GCIP_G3_maxRe_N` | G3 | N | 5940/13365/30260 | `f_window_15_45` | 0.56198 | 0.56181 | 0.56181 | 14.39 | 0.03 | 0.00 | 0.00 | 0.00 | oscillatory |
| `GCIP_G3_maxRe_N` | G3 | N | 5940/13365/30260 | `f_window_30_60` | 0.19924 | 0.19895 | 0.1989 | 4.27 | 0.15 | 0.03 | 0.04 | 0.01 | monotonic |
| `GCIP_G3_maxRe_N` | G3 | N | 5940/13365/30260 | `f_ge_15` | 0.5741 | 0.57396 | 0.57398 | 5.03 | 0.02 | 0.00 | 0.00 | 0.00 | oscillatory |
| `GCIP_G3_maxRe_N` | G3 | N | 5940/13365/30260 | `f_ge_30` | 0.19924 | 0.19895 | 0.1989 | 4.27 | 0.15 | 0.03 | 0.04 | 0.01 | monotonic |
| `GCIP_G3_maxPhi_GN` | G3 | GN | 6156/13851/31300 | `dp_star` | 1943.7 | 1961 | 1976 | 0.35 | 0.88 | 0.76 | 7.15 | 6.15 | monotonic |
| `GCIP_G3_maxPhi_GN` | G3 | GN | 6156/13851/31300 | `xr_over_h1` | 0.022085 | 0.028899 | 0.070988 | - | 23.58 | 59.29 | - | - | divergent |
| `GCIP_G3_maxPhi_GN` | G3 | GN | 6156/13851/31300 | `f_window_15_45` | 0.082311 | 0.082408 | 0.082505 | 0.03 | 0.12 | 0.12 | 13.64 | 13.48 | monotonic |
| `GCIP_G3_maxPhi_GN` | G3 | GN | 6156/13851/31300 | `f_window_30_60` | 0 | 0 | 0 | - | 0.00 | 0.00 | 0.00 | 0.00 | exact |
| `GCIP_G3_maxPhi_GN` | G3 | GN | 6156/13851/31300 | `f_ge_15` | 0.082311 | 0.082408 | 0.082505 | 0.03 | 0.12 | 0.12 | 13.64 | 13.48 | monotonic |
| `GCIP_G3_maxPhi_GN` | G3 | GN | 6156/13851/31300 | `f_ge_30` | 0 | 0 | 0 | - | 0.00 | 0.00 | 0.00 | 0.00 | exact |
| `GCIP_G3_maxPhi_N` | G3 | N | 6156/13851/31300 | `dp_star` | 2012.5 | 2018.3 | 2021.7 | 1.32 | 0.29 | 0.17 | 0.51 | 0.30 | monotonic |
| `GCIP_G3_maxPhi_N` | G3 | N | 6156/13851/31300 | `xr_over_h1` | 0.024499 | 0.034172 | 0.073689 | - | 28.31 | 53.63 | - | - | divergent |
| `GCIP_G3_maxPhi_N` | G3 | N | 6156/13851/31300 | `f_window_15_45` | 0.097684 | 0.097782 | 0.097849 | 0.95 | 0.10 | 0.07 | 0.27 | 0.18 | monotonic |
| `GCIP_G3_maxPhi_N` | G3 | N | 6156/13851/31300 | `f_window_30_60` | 0 | 0 | 0 | - | 0.00 | 0.00 | 0.00 | 0.00 | exact |
| `GCIP_G3_maxPhi_N` | G3 | N | 6156/13851/31300 | `f_ge_15` | 0.097684 | 0.097782 | 0.097849 | 0.95 | 0.10 | 0.07 | 0.27 | 0.18 | monotonic |
| `GCIP_G3_maxPhi_N` | G3 | N | 6156/13851/31300 | `f_ge_30` | 0 | 0 | 0 | - | 0.00 | 0.00 | 0.00 | 0.00 | exact |
| `GCIP_G4_maxRe_GN` | G4 | GN | 3456/7776/17496 | `dp_star` | 300.62 | 303.64 | 305.76 | 0.87 | 0.99 | 0.69 | 2.93 | 2.04 | monotonic |
| `GCIP_G4_maxRe_GN` | G4 | GN | 3456/7776/17496 | `f_window_15_45` | 0.42102 | 0.42028 | 0.42048 | 3.21 | 0.17 | 0.05 | 0.08 | 0.02 | oscillatory |
| `GCIP_G4_maxRe_GN` | G4 | GN | 3456/7776/17496 | `f_window_30_60` | 0.33961 | 0.33917 | 0.37195 | - | 0.13 | 8.81 | - | - | divergent |
| `GCIP_G4_maxRe_GN` | G4 | GN | 3456/7776/17496 | `f_ge_15` | 0.78662 | 0.78706 | 0.78685 | 1.88 | 0.06 | 0.03 | 0.06 | 0.03 | oscillatory |
| `GCIP_G4_maxRe_GN` | G4 | GN | 3456/7776/17496 | `f_ge_30` | 0.54919 | 0.55005 | 0.58238 | - | 0.16 | 5.55 | - | - | divergent |
| `GCIP_G4_maxRe_N` | G4 | N | 3456/7776/17496 | `dp_star` | 272.83 | 274.09 | 274.55 | 2.46 | 0.46 | 0.17 | 0.33 | 0.12 | monotonic |
| `GCIP_G4_maxRe_N` | G4 | N | 3456/7776/17496 | `f_window_15_45` | 0.42103 | 0.4203 | 0.42049 | 3.27 | 0.17 | 0.05 | 0.08 | 0.02 | oscillatory |
| `GCIP_G4_maxRe_N` | G4 | N | 3456/7776/17496 | `f_window_30_60` | 0.3396 | 0.33915 | 0.37194 | - | 0.13 | 8.81 | - | - | divergent |
| `GCIP_G4_maxRe_N` | G4 | N | 3456/7776/17496 | `f_ge_15` | 0.7866 | 0.78703 | 0.78682 | 1.78 | 0.05 | 0.03 | 0.06 | 0.03 | oscillatory |
| `GCIP_G4_maxRe_N` | G4 | N | 3456/7776/17496 | `f_ge_30` | 0.54915 | 0.55001 | 0.58233 | - | 0.16 | 5.55 | - | - | divergent |
| `GCIP_G4_maxPhi_GN` | G4 | GN | 3456/7776/17496 | `dp_star` | 1598.1 | 1615.2 | 1628.2 | 0.68 | 1.06 | 0.80 | 4.20 | 3.17 | monotonic |
| `GCIP_G4_maxPhi_GN` | G4 | GN | 3456/7776/17496 | `f_window_15_45` | 0.45768 | 0.45897 | 0.45857 | 2.88 | 0.28 | 0.09 | 0.16 | 0.05 | oscillatory |
| `GCIP_G4_maxPhi_GN` | G4 | GN | 3456/7776/17496 | `f_window_30_60` | 0.061428 | 0.062202 | 0.076698 | - | 1.24 | 18.90 | - | - | divergent |
| `GCIP_G4_maxPhi_GN` | G4 | GN | 3456/7776/17496 | `f_ge_15` | 0.45768 | 0.45897 | 0.45857 | 2.88 | 0.28 | 0.09 | 0.16 | 0.05 | oscillatory |
| `GCIP_G4_maxPhi_GN` | G4 | GN | 3456/7776/17496 | `f_ge_30` | 0.061428 | 0.062202 | 0.076698 | - | 1.24 | 18.90 | - | - | divergent |
| `GCIP_G4_maxPhi_N` | G4 | N | 3456/7776/17496 | `dp_star` | 1629.6 | 1637 | 1639.9 | 2.31 | 0.45 | 0.18 | 0.36 | 0.14 | monotonic |
| `GCIP_G4_maxPhi_N` | G4 | N | 3456/7776/17496 | `f_window_15_45` | 0.4559 | 0.45716 | 0.48763 | - | 0.28 | 6.25 | - | - | divergent |
| `GCIP_G4_maxPhi_N` | G4 | N | 3456/7776/17496 | `f_window_30_60` | 0.091349 | 0.092393 | 0.092084 | 3.00 | 1.13 | 0.34 | 0.59 | 0.18 | oscillatory |
| `GCIP_G4_maxPhi_N` | G4 | N | 3456/7776/17496 | `f_ge_15` | 0.4559 | 0.45716 | 0.48763 | - | 0.28 | 6.25 | - | - | divergent |
| `GCIP_G4_maxPhi_N` | G4 | N | 3456/7776/17496 | `f_ge_30` | 0.091349 | 0.092393 | 0.092084 | 3.00 | 1.13 | 0.34 | 0.59 | 0.18 | oscillatory |
| `GCIP_G5_maxRe_N` | G5 | N | 3312/7452/16794 | `dp_star` | 915.06 | 919.8 | 921.7 | 2.25 | 0.52 | 0.21 | 0.43 | 0.17 | monotonic |
| `GCIP_G5_maxRe_N` | G5 | N | 3312/7452/16794 | `f_window_15_45` | 0.33273 | 0.3319 | 0.31786 | - | 0.25 | 4.42 | - | - | divergent |
| `GCIP_G5_maxRe_N` | G5 | N | 3312/7452/16794 | `f_window_30_60` | 0.30829 | 0.28006 | 0.28019 | 13.28 | 10.08 | 0.05 | 0.06 | 0.00 | oscillatory |
| `GCIP_G5_maxRe_N` | G5 | N | 3312/7452/16794 | `f_ge_15` | 0.8242 | 0.82448 | 0.82434 | 1.69 | 0.03 | 0.02 | 0.04 | 0.02 | oscillatory |
| `GCIP_G5_maxRe_N` | G5 | N | 3312/7452/16794 | `f_ge_30` | 0.65296 | 0.65375 | 0.6535 | 2.86 | 0.12 | 0.04 | 0.07 | 0.02 | oscillatory |
| `GCIP_G5_maxPhi_GN` | G5 | GN | 3312/7452/16794 | `dp_star` | 2184.7 | 2207.9 | 2226.5 | 0.54 | 1.05 | 0.84 | 5.32 | 4.23 | monotonic |
| `GCIP_G5_maxPhi_GN` | G5 | GN | 3312/7452/16794 | `f_window_15_45` | 0.37253 | 0.36208 | 0.3539 | 0.60 | 2.88 | 2.31 | 13.06 | 10.47 | monotonic |
| `GCIP_G5_maxPhi_GN` | G5 | GN | 3312/7452/16794 | `f_window_30_60` | 0.32978 | 0.31029 | 0.31047 | 11.53 | 6.28 | 0.06 | 0.07 | 0.00 | oscillatory |
| `GCIP_G5_maxPhi_GN` | G5 | GN | 3312/7452/16794 | `f_ge_15` | 0.82407 | 0.81485 | 0.81475 | 11.14 | 1.13 | 0.01 | 0.02 | 0.00 | monotonic |
| `GCIP_G5_maxPhi_GN` | G5 | GN | 3312/7452/16794 | `f_ge_30` | 0.61939 | 0.62024 | 0.62002 | 3.35 | 0.14 | 0.04 | 0.06 | 0.02 | oscillatory |
| `GCIP_G5_maxPhi_N` | G5 | N | 3312/7452/16794 | `dp_star` | 2234.6 | 2246.9 | 2251.9 | 2.20 | 0.55 | 0.22 | 0.47 | 0.19 | monotonic |
| `GCIP_G5_maxPhi_N` | G5 | N | 3312/7452/16794 | `f_window_15_45` | 0.34235 | 0.34146 | 0.34171 | 3.16 | 0.26 | 0.07 | 0.12 | 0.03 | oscillatory |
| `GCIP_G5_maxPhi_N` | G5 | N | 3312/7452/16794 | `f_window_30_60` | 0.30748 | 0.30718 | 0.31653 | - | 0.10 | 2.95 | - | - | divergent |
| `GCIP_G5_maxPhi_N` | G5 | N | 3312/7452/16794 | `f_ge_15` | 0.82355 | 0.82385 | 0.8237 | 1.77 | 0.04 | 0.02 | 0.04 | 0.02 | oscillatory |
| `GCIP_G5_maxPhi_N` | G5 | N | 3312/7452/16794 | `f_ge_30` | 0.64253 | 0.64336 | 0.65229 | - | 0.13 | 1.37 | - | - | divergent |
| `GCIP_G6_maxRe_GN` | G6 | GN | 8784/19764/44442 | `dp_star` | 433.97 | 436.13 | 438.12 | 0.20 | 0.50 | 0.45 | 7.31 | 6.71 | monotonic |
| `GCIP_G6_maxRe_GN` | G6 | GN | 8784/19764/44442 | `f_window_15_45` | 0.41378 | 0.41412 | 0.41419 | 4.15 | 0.08 | 0.02 | 0.02 | 0.00 | monotonic |
| `GCIP_G6_maxRe_GN` | G6 | GN | 8784/19764/44442 | `f_window_30_60` | 0.046149 | 0.05253 | 0.052512 | 14.54 | 12.15 | 0.03 | 0.04 | 0.00 | oscillatory |
| `GCIP_G6_maxRe_GN` | G6 | GN | 8784/19764/44442 | `f_ge_15` | 0.41378 | 0.41412 | 0.41419 | 4.15 | 0.08 | 0.02 | 0.02 | 0.00 | monotonic |
| `GCIP_G6_maxRe_GN` | G6 | GN | 8784/19764/44442 | `f_ge_30` | 0.046149 | 0.05253 | 0.052512 | 14.54 | 12.15 | 0.03 | 0.04 | 0.00 | oscillatory |
| `GCIP_G6_maxRe_N` | G6 | N | 8784/19764/44442 | `dp_star` | 406.51 | 407.05 | 407.58 | 0.02 | 0.13 | 0.13 | 22.35 | 22.16 | monotonic |
| `GCIP_G6_maxRe_N` | G6 | N | 8784/19764/44442 | `f_window_15_45` | 0.41377 | 0.41411 | 0.41417 | 4.49 | 0.08 | 0.01 | 0.02 | 0.00 | monotonic |
| `GCIP_G6_maxRe_N` | G6 | N | 8784/19764/44442 | `f_window_30_60` | 0.052295 | 0.052527 | 0.052508 | 6.10 | 0.44 | 0.04 | 0.05 | 0.00 | oscillatory |
| `GCIP_G6_maxRe_N` | G6 | N | 8784/19764/44442 | `f_ge_15` | 0.41377 | 0.41411 | 0.41417 | 4.49 | 0.08 | 0.01 | 0.02 | 0.00 | monotonic |
| `GCIP_G6_maxRe_N` | G6 | N | 8784/19764/44442 | `f_ge_30` | 0.052295 | 0.052527 | 0.052508 | 6.10 | 0.44 | 0.04 | 0.05 | 0.00 | oscillatory |
| `GCIP_G6_maxPhi_GN` | G6 | GN | 22608/50868/114210 | `dp_star` | 6943.5 | 6971.9 | 6995.6 | 0.44 | 0.41 | 0.34 | 2.60 | 2.17 | monotonic |
| `GCIP_G6_maxPhi_GN` | G6 | GN | 22608/50868/114210 | `f_window_15_45` | 0.5424 | 0.54271 | 0.54276 | 4.27 | 0.06 | 0.01 | 0.02 | 0.00 | monotonic |
| `GCIP_G6_maxPhi_GN` | G6 | GN | 22608/50868/114210 | `f_window_30_60` | 0.15561 | 0.15597 | 0.15597 | 16.86 | 0.23 | 0.00 | 0.00 | 0.00 | oscillatory |
| `GCIP_G6_maxPhi_GN` | G6 | GN | 22608/50868/114210 | `f_ge_15` | 0.5424 | 0.54271 | 0.54276 | 4.27 | 0.06 | 0.01 | 0.02 | 0.00 | monotonic |
| `GCIP_G6_maxPhi_GN` | G6 | GN | 22608/50868/114210 | `f_ge_30` | 0.15561 | 0.15597 | 0.15597 | 16.86 | 0.23 | 0.00 | 0.00 | 0.00 | oscillatory |
| `GCIP_G6_maxPhi_N` | G6 | N | 22608/50868/114210 | `dp_star` | 7212.4 | 7225.1 | 7235.5 | 0.51 | 0.18 | 0.14 | 0.96 | 0.78 | monotonic |
| `GCIP_G6_maxPhi_N` | G6 | N | 22608/50868/114210 | `f_window_15_45` | 0.53456 | 0.53478 | 0.53485 | 2.89 | 0.04 | 0.01 | 0.02 | 0.01 | monotonic |
| `GCIP_G6_maxPhi_N` | G6 | N | 22608/50868/114210 | `f_window_30_60` | 0.19842 | 0.19881 | 0.19883 | 8.22 | 0.20 | 0.01 | 0.01 | 0.00 | monotonic |
| `GCIP_G6_maxPhi_N` | G6 | N | 22608/50868/114210 | `f_ge_15` | 0.54047 | 0.54077 | 0.54083 | 3.63 | 0.05 | 0.01 | 0.02 | 0.00 | monotonic |
| `GCIP_G6_maxPhi_N` | G6 | N | 22608/50868/114210 | `f_ge_30` | 0.19842 | 0.19881 | 0.19883 | 8.22 | 0.20 | 0.01 | 0.01 | 0.00 | monotonic |
| `GCIP_G7_maxRe_GN` | G7 | GN | 13320/29970/66920 | `dp_star` | 379.38 | 385.05 | 390.77 | - | 1.47 | 1.47 | - | - | divergent |
| `GCIP_G7_maxRe_GN` | G7 | GN | 13320/29970/66920 | `xr_over_h1` | 1.4783 | 1.4862 | 1.494 | 0.05 | 0.54 | 0.52 | 30.08 | 29.27 | monotonic |
| `GCIP_G7_maxRe_GN` | G7 | GN | 13320/29970/66920 | `f_window_15_45` | 0 | 0 | 0 | - | 0.00 | 0.00 | 0.00 | 0.00 | exact |
| `GCIP_G7_maxRe_GN` | G7 | GN | 13320/29970/66920 | `f_window_30_60` | 0 | 0 | 0 | - | 0.00 | 0.00 | 0.00 | 0.00 | exact |
| `GCIP_G7_maxRe_GN` | G7 | GN | 13320/29970/66920 | `f_ge_15` | 0 | 0 | 0 | - | 0.00 | 0.00 | 0.00 | 0.00 | exact |
| `GCIP_G7_maxRe_GN` | G7 | GN | 13320/29970/66920 | `f_ge_30` | 0 | 0 | 0 | - | 0.00 | 0.00 | 0.00 | 0.00 | exact |
| `GCIP_G7_maxRe_N` | G7 | N | 13320/29970/66920 | `dp_star` | 337.52 | 337.71 | 337.82 | 1.16 | 0.05 | 0.03 | 0.11 | 0.07 | monotonic |
| `GCIP_G7_maxRe_N` | G7 | N | 13320/29970/66920 | `xr_over_h1` | 1.5055 | 1.5249 | 1.5224 | 5.06 | 1.27 | 0.16 | 0.23 | 0.03 | oscillatory |
| `GCIP_G7_maxRe_N` | G7 | N | 13320/29970/66920 | `f_window_15_45` | 0 | 0 | 0 | - | 0.00 | 0.00 | 0.00 | 0.00 | exact |
| `GCIP_G7_maxRe_N` | G7 | N | 13320/29970/66920 | `f_window_30_60` | 0 | 0 | 0 | - | 0.00 | 0.00 | 0.00 | 0.00 | exact |
| `GCIP_G7_maxRe_N` | G7 | N | 13320/29970/66920 | `f_ge_15` | 0 | 0 | 0 | - | 0.00 | 0.00 | 0.00 | 0.00 | exact |
| `GCIP_G7_maxRe_N` | G7 | N | 13320/29970/66920 | `f_ge_30` | 0 | 0 | 0 | - | 0.00 | 0.00 | 0.00 | 0.00 | exact |
| `GCIP_G7_maxPhi_GN` | G7 | GN | 19152/43092/97038 | `dp_star` | 660.58 | 671.44 | 682.09 | 0.05 | 1.62 | 1.56 | 98.36 | 94.88 | monotonic |
| `GCIP_G7_maxPhi_GN` | G7 | GN | 19152/43092/97038 | `xr_over_h1` | 0.50988 | 0.53523 | 0.53434 | 8.26 | 4.74 | 0.17 | 0.22 | 0.01 | oscillatory |
| `GCIP_G7_maxPhi_GN` | G7 | GN | 19152/43092/97038 | `f_window_15_45` | 0 | 0 | 0 | - | 0.00 | 0.00 | 0.00 | 0.00 | exact |
| `GCIP_G7_maxPhi_GN` | G7 | GN | 19152/43092/97038 | `f_window_30_60` | 0 | 0 | 0 | - | 0.00 | 0.00 | 0.00 | 0.00 | exact |
| `GCIP_G7_maxPhi_GN` | G7 | GN | 19152/43092/97038 | `f_ge_15` | 0 | 0 | 0 | - | 0.00 | 0.00 | 0.00 | 0.00 | exact |
| `GCIP_G7_maxPhi_GN` | G7 | GN | 19152/43092/97038 | `f_ge_30` | 0 | 0 | 0 | - | 0.00 | 0.00 | 0.00 | 0.00 | exact |
| `GCIP_G7_maxPhi_N` | G7 | N | 19152/43092/97038 | `dp_star` | 613.12 | 613.51 | 613.67 | 2.32 | 0.06 | 0.03 | 0.05 | 0.02 | monotonic |
| `GCIP_G7_maxPhi_N` | G7 | N | 19152/43092/97038 | `xr_over_h1` | 0.6553 | 0.66718 | 0.66476 | 3.92 | 1.78 | 0.36 | 0.57 | 0.12 | oscillatory |
| `GCIP_G7_maxPhi_N` | G7 | N | 19152/43092/97038 | `f_window_15_45` | 0 | 0 | 0 | - | 0.00 | 0.00 | 0.00 | 0.00 | exact |
| `GCIP_G7_maxPhi_N` | G7 | N | 19152/43092/97038 | `f_window_30_60` | 0 | 0 | 0 | - | 0.00 | 0.00 | 0.00 | 0.00 | exact |
| `GCIP_G7_maxPhi_N` | G7 | N | 19152/43092/97038 | `f_ge_15` | 0 | 0 | 0 | - | 0.00 | 0.00 | 0.00 | 0.00 | exact |
| `GCIP_G7_maxPhi_N` | G7 | N | 19152/43092/97038 | `f_ge_30` | 0 | 0 | 0 | - | 0.00 | 0.00 | 0.00 | 0.00 | exact |
| `GCIP_G5_maxRe_GN` | G5 | GN | 3312/7452/FAIL | `dp_star` | 943.37 | 952.34 | — (L3 `not_converged`) | - | 0.94 | - | - | - | incomplete |
| `GCIP_G5_maxRe_GN` | G5 | GN | 3312/7452/FAIL | `f_window_15_45` | 0.33259 | 0.33177 | — (L3 `not_converged`) | - | 0.25 | - | - | - | incomplete |
| `GCIP_G5_maxRe_GN` | G5 | GN | 3312/7452/FAIL | `f_window_30_60` | 0.30828 | 0.30801 | — (L3 `not_converged`) | - | 0.09 | - | - | - | incomplete |
| `GCIP_G5_maxRe_GN` | G5 | GN | 3312/7452/FAIL | `f_ge_15` | 0.82435 | 0.82465 | — (L3 `not_converged`) | - | 0.04 | - | - | - | incomplete |
| `GCIP_G5_maxRe_GN` | G5 | GN | 3312/7452/FAIL | `f_ge_30` | 0.65322 | 0.65402 | — (L3 `not_converged`) | - | 0.12 | - | - | - | incomplete |

### Family-level GCI verdict (`GCI_fine > 5 %` rule on `dp_star` or `xr_over_h1`)

| Family | Max `dp*` `e21` N / GN (%) | Max `dp*` `GCI_fine` N / GN (%) | Max `x_r` `e21` (%) | Max `x_r` `GCI_fine` (%) | Rule verdict | Root cause / note |
|---|---|---|---|---|---|---|
| **G2** | 0.08 / 1.50 | 1.04 / **234.94** | 0.25 | 0.00 | **`needs_level3`** (literal) | V1 `N` passes (`≤1.04 %`); V2 `GN` `dp*` grows ~1.4 %/level with `p ≈ 0.02–0.10` (uniform plug-flow inlet corner singularity, FLAG 1); `x_r` converged (`0.00 %`) |
| **G3** | 0.17 / 0.76 | 0.30 / **6.93** | 59.29 (`maxPhi`) / 2.42 (`maxRe`) | `divergent` (all 4) | **`needs_level3`** (literal + `x_r` div.) | V2 `GN` `dp*` `6.15–6.93 %` (FLAG 1); `x_r` divergent on all 4 cases (`maxPhi` tiny corner bubble `0.022→0.029→0.071`; `maxRe` `0.255→0.261→0.267`, FLAG 2) |
| **G4** | 0.18 / 0.80 | 0.14 / 3.17 | — | — | **PASS** (`dp* ≤ 3.17 %`) | All `dp*` `GCI_fine ≤ 3.17 %`; note discrete `f_window_30_60` jumps (`e21` 8.8–18.9 %, divergent) on 80-pt outlet sample line (FLAG 3) |
| **G5** | 0.22 / 0.84* | 0.19 / 4.23* | — | — | **PASS (caveat: 1 L3 fail)** | Complete 3 cases pass (`≤4.23 %`); `GCIP_G5_maxRe_GN_L3` failed `not_converged` (`e32 = 0.94 %`); `G5_maxPhi_GN` `f_window_15_45` `GCI_fine = 10.47 %` (`e21 = 2.31 %`) |
| **G6** | 0.14 / 0.45 | **22.16** / **6.71** | — | — | **`needs_level3`** (literal) | `e21` is tiny (`0.13–0.45 %`), inflated by `p = 0.02` (`G6_maxRe_N`: `406.51→407.05→407.58`) and `p = 0.20` (`G6_maxRe_GN`, FLAG 1) |
| **G7** | 0.03 / 1.56 | 0.07 / **94.88** (`maxRe_GN` div.) | 0.52 | **29.27** (`maxRe_GN`) | **`needs_level3`** (literal) | V1 `N` passes (`≤0.12 %`); V2 `GN` `dp*` grows 1.5 %/level (FLAG 1); `G7_maxRe_GN` `x_r` `1.478→1.486→1.494` (`e21 = 0.52 %`, `p = 0.05` → `29.27 %`) |

### Diagnostic FLAGs for the team

1. **FLAG 1 — Uniform-inlet corner pressure singularity explains V2 (`GN`) `dp*` low observed order (`p ≈ 0.02–0.87`) across G2–G7.**
   Every V1 (`N`, parabolic `exprFixedValue` inlet `6y(1-y)`) case has direct L2→L3 `dp*` change `e21 ≤ 0.22 %` (`p ≈ 1.2–2.5` on 10/12 cases).
   In contrast, `cfd/gen/case.py:72-81` prescribes `fixedValue uniform (1 0 0)` at the inlet for all `BirdCarreau` (`GN`) cases, creating a velocity discontinuity at the inlet-wall corners `(-L_in, 0)` and `(-L_in, 1)`.
   Because `campaign/run_batch.py::_compute_dp_star` evaluates `p_inlet` at the first two cell columns next to `x = -L_in` and extrapolates to `x_in`, each 1.5× refinement resolves a sharper entrance pressure spike (`~0.4–1.6 %` constant increment per level ⇒ `e32 ≈ e21` ⇒ `p → 0` ⇒ Roache `GCI_fine` explodes to `6–235 %`).
   **Implication:** Re-running V2 at level 3 will NOT remove this logarithmic entrance-corner growth. The proper fix is either (a) evaluating `dp*` from a developed inlet plane (e.g. `x* = -1`) in post-processing from the saved `.npz`/archives, or (b) prescribing a developed 1D Carreau inlet profile (`CORPUS_PLAN.md` §6.1).
2. **FLAG 2 — G3 stenosis `x_r/h₁` is not mesh-converged at L2/L3 for tiny bubbles (`maxPhi`) and grows slowly at `maxRe`.**
   In `G3_maxPhi` (`Re = 6.74`, `throat = 0.664`), `x_r/h₁` jumps `0.022 → 0.029 → 0.071` (`N`: `0.024 → 0.034 → 0.074`) because the separation bubble is only a few cells long on `L1/L2` (`nearWallDownstream` sits at fixed distance from the wall). At `G3_maxRe` (`Re = 51.51`), `x_r/h₁` grows steadily `0.255 → 0.261 → 0.267` (`e21 = 2.4 %`, `divergent`).
3. **FLAG 3 — Outlet `f_window` discrete step jumps on bifurcations (G4/G5) and zero-window corners.**
   `f_window` integrates an indicator `1[lo ≤ τ ≤ hi]` over 80 discrete sample points across the daughter outlets; when a threshold boundary (`15` or `30 Pa`) crosses a sample point between L2 and L3, `f_window_30_60` jumps by 1–2 quadrature bins (`G4_maxRe`: `0.339 → 0.372`, `G4_maxPhi_GN`: `0.062 → 0.077`). Conversely, for expansion corners whose expanded outlet has `τ_wall < 15 Pa` (`G2_maxPhi`, `G7_maxPhi`, `G7_maxRe`), outlet `f_window` is identically `0` on all 3 levels (high shear occurs in the inlet/throat/step, not the expanded outlet).
4. **FLAG 4 — `GCIP_G5_maxRe_GN_L3` (`B008`, Re = 52.43, φ = 0.3837) failed `not_converged` even after `p = 0.2, U = 0.5` retry.**
   Final state at 5000 it: `Ux = 1.81e-6`, `Uy = 2.72e-6`, `p = 8.55e-5`, `continuity_error = 1.79e-7`, `dp_rel_change_500 = 4.25e-6`, `u_out_rel_change_500 = 2.57e-4`. Archived in GCS.

### `needs_level3` run_ids (T10 step 8 literal rule: families G2, G3, G6, G7)

Listed for the team decision (NOT re-run). Note that under FLAG 1, G2/G6/G7 `dp*` flags stem from the V2 uniform-inlet corner singularity (`e21 ≤ 1.56 %`), while V1 `N` has `e21 ≤ 0.22 %` across all families.

- **G2 (54 V1/V2 run_ids):** `P_G2_B000_GN`, `P_G2_B000_N`, `P_G2_B001_GN`, `P_G2_B001_N`, `P_G2_B002_GN`, `P_G2_B002_N`, `P_G2_B003_GN`, `P_G2_B003_N`, `P_G2_B004_GN`, `P_G2_B004_N`, `P_G2_B005_GN`, `P_G2_B005_N`, `P_G2_B006_GN`, `P_G2_B006_N`, `P_G2_B007_GN`, `P_G2_B007_N`, `P_G2_B008_GN`, `P_G2_B008_N`, `P_G2_B009_GN`, `P_G2_B009_N`, `P_G2_B010_GN`, `P_G2_B010_N`, `P_G2_B011_GN`, `P_G2_B011_N`, `P_G2_C000_GN`, `P_G2_C000_N`, `P_G2_C001_GN`, `P_G2_C001_N`, `P_G2_C002_GN`, `P_G2_C002_N`, `P_G2_D000_GN`, `P_G2_D000_N`, `P_G2_D001_GN`, `P_G2_D001_N`, `P_G2_D002_GN`, `P_G2_D002_N`, `P_G2_D003_GN`, `P_G2_D003_N`, `P_G2_D004_GN`, `P_G2_D004_N`, `P_G2_D005_GN`, `P_G2_D005_N`, `P_G2_D006_GN`, `P_G2_D006_N`, `P_G2_D007_GN`, `P_G2_D007_N`, `P_G2_D008_GN`, `P_G2_D008_N`, `P_G2_D009_GN`, `P_G2_D009_N`, `P_G2_D010_GN`, `P_G2_D010_N`, `P_G2_D011_GN`, `P_G2_D011_N`
- **G3 (54 V1/V2 run_ids):** `P_G3_B000_GN`, `P_G3_B000_N`, `P_G3_B001_GN`, `P_G3_B001_N`, `P_G3_B002_GN`, `P_G3_B002_N`, `P_G3_B003_GN`, `P_G3_B003_N`, `P_G3_B004_GN`, `P_G3_B004_N`, `P_G3_B005_GN`, `P_G3_B005_N`, `P_G3_B006_GN`, `P_G3_B006_N`, `P_G3_B007_GN`, `P_G3_B007_N`, `P_G3_B008_GN`, `P_G3_B008_N`, `P_G3_B009_GN`, `P_G3_B009_N`, `P_G3_B010_GN`, `P_G3_B010_N`, `P_G3_B011_GN`, `P_G3_B011_N`, `P_G3_C000_GN`, `P_G3_C000_N`, `P_G3_C001_GN`, `P_G3_C001_N`, `P_G3_C002_GN`, `P_G3_C002_N`, `P_G3_D000_GN`, `P_G3_D000_N`, `P_G3_D001_GN`, `P_G3_D001_N`, `P_G3_D002_GN`, `P_G3_D002_N`, `P_G3_D003_GN`, `P_G3_D003_N`, `P_G3_D004_GN`, `P_G3_D004_N`, `P_G3_D005_GN`, `P_G3_D005_N`, `P_G3_D006_GN`, `P_G3_D006_N`, `P_G3_D007_GN`, `P_G3_D007_N`, `P_G3_D008_GN`, `P_G3_D008_N`, `P_G3_D009_GN`, `P_G3_D009_N`, `P_G3_D010_GN`, `P_G3_D010_N`, `P_G3_D011_GN`, `P_G3_D011_N`
- **G6 (54 V1/V2 run_ids):** `P_G6_B000_GN`, `P_G6_B000_N`, `P_G6_B001_GN`, `P_G6_B001_N`, `P_G6_B002_GN`, `P_G6_B002_N`, `P_G6_B003_GN`, `P_G6_B003_N`, `P_G6_B004_GN`, `P_G6_B004_N`, `P_G6_B005_GN`, `P_G6_B005_N`, `P_G6_B006_GN`, `P_G6_B006_N`, `P_G6_B007_GN`, `P_G6_B007_N`, `P_G6_B008_GN`, `P_G6_B008_N`, `P_G6_B009_GN`, `P_G6_B009_N`, `P_G6_B010_GN`, `P_G6_B010_N`, `P_G6_B011_GN`, `P_G6_B011_N`, `P_G6_C000_GN`, `P_G6_C000_N`, `P_G6_C001_GN`, `P_G6_C001_N`, `P_G6_C002_GN`, `P_G6_C002_N`, `P_G6_D000_GN`, `P_G6_D000_N`, `P_G6_D001_GN`, `P_G6_D001_N`, `P_G6_D002_GN`, `P_G6_D002_N`, `P_G6_D003_GN`, `P_G6_D003_N`, `P_G6_D004_GN`, `P_G6_D004_N`, `P_G6_D005_GN`, `P_G6_D005_N`, `P_G6_D006_GN`, `P_G6_D006_N`, `P_G6_D007_GN`, `P_G6_D007_N`, `P_G6_D008_GN`, `P_G6_D008_N`, `P_G6_D009_GN`, `P_G6_D009_N`, `P_G6_D010_GN`, `P_G6_D010_N`, `P_G6_D011_GN`, `P_G6_D011_N`
- **G7 (54 V1/V2 run_ids):** `P_G7_B000_GN`, `P_G7_B000_N`, `P_G7_B001_GN`, `P_G7_B001_N`, `P_G7_B002_GN`, `P_G7_B002_N`, `P_G7_B003_GN`, `P_G7_B003_N`, `P_G7_B004_GN`, `P_G7_B004_N`, `P_G7_B005_GN`, `P_G7_B005_N`, `P_G7_B006_GN`, `P_G7_B006_N`, `P_G7_B007_GN`, `P_G7_B007_N`, `P_G7_B008_GN`, `P_G7_B008_N`, `P_G7_B009_GN`, `P_G7_B009_N`, `P_G7_B010_GN`, `P_G7_B010_N`, `P_G7_B011_GN`, `P_G7_B011_N`, `P_G7_C000_GN`, `P_G7_C000_N`, `P_G7_C001_GN`, `P_G7_C001_N`, `P_G7_C002_GN`, `P_G7_C002_N`, `P_G7_D000_GN`, `P_G7_D000_N`, `P_G7_D001_GN`, `P_G7_D001_N`, `P_G7_D002_GN`, `P_G7_D002_N`, `P_G7_D003_GN`, `P_G7_D003_N`, `P_G7_D004_GN`, `P_G7_D004_N`, `P_G7_D005_GN`, `P_G7_D005_N`, `P_G7_D006_GN`, `P_G7_D006_N`, `P_G7_D007_GN`, `P_G7_D007_N`, `P_G7_D008_GN`, `P_G7_D008_N`, `P_G7_D009_GN`, `P_G7_D009_N`, `P_G7_D010_GN`, `P_G7_D010_N`, `P_G7_D011_GN`, `P_G7_D011_N`
