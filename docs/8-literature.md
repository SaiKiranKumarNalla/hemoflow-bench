# 8 · Literature and references

Literature review and the reference list.

## Contents

- [Literature (checked 2026-11-08)](#literature-checked-2026-11-08)
- [Published Euler–Euler (two-fluid) RBC studies relevant to M2b](#published-eulereuler-two-fluid-rbc-studies-relevant-to-m2b)
- [REFERENCES](#references)

---

## Literature (checked 2026-11-08)


| Ref | What they did | Relevance |
|---|---|---|
| [Piergiovanni et al. 2020, Ann Biomed Eng 48:236](https://link.springer.com/article/10.1007/s10439-019-02342-w) | Shear-induced encapsulation in a straight microchannel; CFD + micro-PIV; FITC-dextran 40 kDa; defines index I | Core method and index |
| [Piergiovanni-group follow-up 2025, Ann Biomed Eng](https://pmc.ncbi.nlm.nih.gov/articles/PMC12006216) | Haematocrit and albumin effects; 70–94 % of cells loaded; species differences | Blood composition matters → patient inputs |
| [Controlled shear on the erythrocyte membrane (PubMed 26813612)](https://pubmed.ncbi.nlm.nih.gov/26813612/) | Earlier controlled-shear encapsulation study | Origin of the approach |
| [Rahman et al. 2021, Lab Chip 21:2121](https://pubs.rsc.org/en/Content/ArticleLanding/2021/LC/D0LC01077D) | Protein loading into RBCs via transient vs extended shear-mediated deformation | One device design, no optimisation |
| [Gupta et al. 2026, Lab Chip 26:3825](https://pmc.ncbi.nlm.nih.gov/articles/PMC13208450/) | Parametric mechanoporation design study (channel count/width, flow 100–350 µL/min, adhesion, stiffness); 3D COMSOL + experiments; strain rate and history forces predict delivery | **Closest prior art** — but melanoma cells and T-cells, compression-based, no surrogate, no per-donor optimisation |
| [Chakrabarty et al. 2025, Adv Healthc Mater](https://advanced.onlinelibrary.wiley.com/doi/10.1002/adhm.202502472) | Cell-squeezing device; deep learning for result analysis | ML for readout, not a physics surrogate |
| [Wu et al. 2015, Int J Eng Sci 95:49](https://pmc.ncbi.nlm.nih.gov/articles/PMC4521229/) | Two-fluid blood in benchmark micro-channels (OpenFOAM) | Validation benchmark |
| [Euler–Euler sudden expansion, 2026](https://pmc.ncbi.nlm.nih.gov/articles/PMC13029181/) | EE with rigid-sphere RBCs + modified Carreau–Yasuda | Validation / model choice |
| [Core–plasma two-phase model, 2025](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC12758828/) | Two-phase core + cell-free layer; compares hemodynamic models | Model-choice reference |
| [EE blood in oxygenators, 2024](https://pubmed.ncbi.nlm.nih.gov/39531093/) | EE calibrated on microfluidics, applied to fibre bundles | Calibration practice |
| [Cell-free-layer CFD validation, 2022](https://pubmed.ncbi.nlm.nih.gov/36122496/) | EE vs Euler–Lagrange in a hyperbolic contraction | Validation metric: CFL thickness |
| [Sohrabi & Liu 2017, Artif Organs](https://doi.org/10.1111/aor.12832) | Cellular model of shear-induced hemolysis | Damage threshold |
| [Hemolysis DNN surrogate, FDA nozzle (COMSOL 2024)](https://www.comsol.com/paper/deep-neural-network-surrogate-model-for-blood-damage-modeling-in-fda-hemolysis-benchmark-136422) | NN surrogate for power-law hemolysis; ~10 % error with log target | Surrogate precedent (single-phase) |
| [Neural operator for platelets, 2025](https://arxiv.org/pdf/2503.12074) | DeepONet for platelet deformation in shear | Neural-operator precedent in blood |
| [Physics-guided surrogate for DLD design, 2025](https://arxiv.org/pdf/2512.05649) | Surrogate-driven microfluidic design (cell sorting) | Method precedent for microfluidic inverse design |
| [Multi-scale RBC trauma in high-shear flows, 2024](https://arxiv.org/pdf/2411.13002) | Couples continuum CFD with cell-level damage | Template for L3 calibration |
| [HemoCell](https://arxiv.org/pdf/2305.02752) | Open-source cell-resolved blood flow (LBM + immersed boundary) | L3 tool; no pore model |
| [Multi-fidelity FNO with transfer learning](https://arxiv.org/pdf/2304.06972), [multi-fidelity neural operators](https://arxiv.org/html/2608.04708v1), [multi-fidelity scaling laws](https://arxiv.org/pdf/2511.01830) | Pretrain on low fidelity, fine-tune on high fidelity | S2 method |
| [Blood rheology review 2026](https://arxiv.org/pdf/2609.20855), [rheology in the aorta](https://pmc.ncbi.nlm.nih.gov/articles/PMC10669506/), [RBC aggregation and viscosity](https://journals.plos.org/plosone/article?id=10.1371%2Fjournal.pone.0199911) | Rheology model choice | 04-blood-models |

### Gaps (Stage A)
1. **Geometry/flow optimisation for RBC shear encapsulation** — done for other cells under compression (Gupta 2026),
   not for RBCs under shear.
2. **Patient/donor variability** — composition changes loading (2025 follow-up), but no model chooses conditions per donor.
3. **Throughput vs encapsulation vs hemolysis trade-off** — no Pareto front.
4. **No physics surrogate** for encapsulation (ML so far only reads out images).
5. Out of scope: mechanistic pore kinetics linked to continuum models; full 3D device effects beyond a small L2 set.

`TODO (agent): before claiming novelty, search Google Scholar for papers citing Piergiovanni 2020 and Rahman 2021.`

---

## Published Euler–Euler (two-fluid) RBC studies relevant to M2b


Read 2026-11-24. Wu 2015 methods/results were read verbatim; Sezer 2026 was
read verbatim up to §2.3 (model, closures, mesh), its results only via a summary → marked `secondary` below.

| | Wu et al. 2015 | Sezer et al. 2026 |
|---|---|---|
| Citation | Wu W-T, Yang F, Antaki JF, Aubry N, Massoudi M. *Int J Eng Sci* 95:49–59. doi:10.1016/j.ijengsci.2015.06.004 (PMC4521229) | Sezer C, Kaya K, Tabatabaei Malazi M, Dalkılıç AS. *Micromachines* 17(3):316 (2026). doi:10.3390/mi17030316 (PMC13029181) |
| Solver | **custom** two-fluid solver "developed using the solvers and libraries of OpenFOAM"; base solver and version not stated | ANSYS Fluent 2023R2, steady, coupled p–U, PRESTO, 1st-order upwind α, 2nd-order momentum |
| Drag | Stokes drag (with hindrance, see [§2-theory](2-theory.md)) | Gidaspow: Wen–Yu (α_R < 0.2), Ergun (α_R ≥ 0.2) |
| Lift | Saffman shear lift | Saffman–Mei |
| RBC phase | Yeleswarapu shear-thinning viscoelastic; a = 8 µm | mixture Carreau–Yasuda (Jung et al.), λ = 0.110 s, m(α), n(α) polynomials split at γ̇ = 6 s⁻¹; μ_R back-calculated |
| Properties | μ_f 0.96 cP, ρ_f 1027, ρ_rbc 1093 kg/m³ | ρ_p 1027, ρ_R 1093 kg/m³, μ_p 0.96 mPa·s |
| Cases | Karino & Goldsmith tube (Re 12.2/37.8), Patrick channel (Hct 0.45), Zhao expansion (Hct 0.20), crevices (Hct 0.40) | Zhao expansion 100→200 µm (L 1000/2000 µm), α_R 0.2–0.5, d 4/8/11 µm, 0.6/50/100 µL/min (Re 0.11/8.9/17.8) |
| Mesh | ICEM hex; Zhao converged at 125 544 nodes | 0.8/1.6/3.2 M hex; Δp change 0.02 % medium→fine |
| Quantitative results | Karino reattachment (experimental 230/715 µm); others qualitative; **no CFL numbers** | ΔP* = 226.72 α_R² − 148.57 α_R + 117.37 (max err 1.13 %), min at α_R ≈ 0.3; CFL thicker for larger d, thinner for higher α_R — `secondary` (summary) |
| Code / data | not available | not available; data on request (`secondary`) |

**Consequences for M2b:** both studies use a Saffman-type lift → M2b-B (`reactingTwoPhaseEulerFoam`
+ `LegendreMagnaudet`, [§2-theory](2-theory.md) §6) is the physically comparable variant; M2b-A (no lift) is the control.
Stock ESI `GidaspowErgunWenYu` drag reproduces Sezer's drag law. The Sezer ΔP*(α_R) correlation is a quantitative
two-fluid target (verify Eq. numbers and the ΔP* definition in the full PDF before using it).

**Update 2026-11-24: Sezer 2026 full text read (Europe PMC XML, PMC13029181) → `verified`.** ΔP* = ΔP/(½ρ_p u_i²)
(Eq. 22); ΔP* = 226.72 α_R² − 148.57 α_R + 117.37 (Eq. 23, max rel. error 1.13 % at α_R 0.2, d 11 µm), minimum at
α_R ≈ 0.3; Gidaspow drag Eq. 14a/b (Ergun α_R ≥ 0.2, Wen–Yu < 0.2), Saffman–Mei lift Eqs. 17–19, Jung Carreau–Yasuda
mixture viscosity Eqs. 5–11 (λ = 0.110 s, split at γ̇ = 6 s⁻¹), μ_R back-calculated Eq. 11; 3D quarter domain,
h₁ = W = 100 µm, h₂ = 200 µm, L₁ 1000 / L₂ 2000 µm; uniform inlet velocity and φ; mesh 0.8/1.6/3.2 M (ΔP 0.65 % /
0.02 %); CFL = first inflection point from the wall, |y/h| < 0.48 (Eq. 21); validation vs Patrick et al. RBC velocity
(Fig. 7) and Kim et al. φ (Fig. 8). CFL values are given in figures only (Fig. 13). See [§4-validation](4-validation.md).

---

## REFERENCES


Only references that already appear in `knowledge/` are listed, with the DOI / PMCID / PMID / arXiv
identifier exactly as recorded there. Verification labels follow `knowledge/`: `verified-primary` = original
full text read; `secondary` = value quoted by a paper we read, original not read; `abstract` = abstract only.
Works that `knowledge/` names only inside citing papers (e.g. Evans 1976, Saffman 1965, Rusche & Issa,
Yeleswarapu, Brooks, Jung & Hassanein, Pries/Fåhræus, Karino & Goldsmith 1977, Secomb & Pries, Recktenwald,
ICSH 1984, Quemada 1976/77, Legendre & Magnaudet 1998, Olla 2000) are **not** standalone entries here — they
are cited with `secondary`/`recalled` labels where used.

### Application: RBC drug carriers, shear encapsulation, release, hemolysis

1. Piergiovanni M, Casagrande G, Taverna F, et al. Shear-induced encapsulation into red blood cells: a new
   microfluidic approach to drug delivery. *Ann Biomed Eng* 48:236–246 (2020).
   doi:10.1007/s10439-019-02342-w
2. Bernardelli N, Piergiovanni M, Bianchi V, Carlo-Stella C, Costantino L, Casagrande C. *Ann Biomed Eng*
   (2025). doi:10.1007/s10439-025-03678-2 · PMCID PMC12006216 (`verified`, full text read)
3. Controlled shear on the erythrocyte membrane. PMID 26813612
   (pubmed.ncbi.nlm.nih.gov/26813612, as listed in [§8-literature](8-literature.md))
4. Rahman MH, Wong CHN, Lee MM, Chan MK, Ho YP. Efficient encapsulation of functional proteins into
   erythrocytes by controlled shear-mediated membrane deformation. *Lab Chip* 21:2121–2128 (2021).
   pubs.rsc.org/en/Content/ArticleLanding/2021/LC/D0LC01077D
5. Glassman PM et al. Red blood cells: the metamorphosis of a neglected carrier… *Adv Drug Deliv Rev*
   178:113992 (2021). doi:10.1016/j.addr.2021.113992 (ref [1] of the 2023 report)
6. Koleva L, Bovt E, Ataullakhanov F, Sinauridze E. Erythrocytes as carriers: from drug delivery to
   biosensors. *Pharmaceutics* 12(3):276 (2020). doi:10.3390/pharmaceutics12030276
7. Glassman PM et al. Vascular drug delivery using carrier red blood cells. *Pharmaceutics* 12(5):440
   (2020). doi:10.3390/pharmaceutics12050440
8. Bao N, Le TT, Cheng JX, Lu C. Microfluidic electroporation of tumor and blood cells. *Integr Biol*
   2(2–3):83–90 (2010). doi:10.1039/b919820b
9. Sohrabi S, Liu Y. A cellular model of shear-induced hemolysis. *Artif Organs* 41(9):E80–E91 (2017).
   doi:10.1111/aor.12832
10. Amar et al. Shear-triggered release. *Small* (2026). doi:10.1002/smll.74818 · PMCID PMC13580257
    (`verified`, full text read)
11. Zeibi Shirejini H, Carberry T, Alt A, Gregory J, Hagemeyer A. Shear-responsive delivery review.
    *Adv Funct Mater* (2023). doi:10.1002/adfm.202303717 (metadata only)
12. Park et al. RBC hitchhiking in vivo. *Nat Commun* (2025). doi:10.1038/s41467-025-65185-1 ·
    PMCID PMC12638801 (`abstract`; background only)
13. Gupta et al. Parametric mechanoporation design study. *Lab Chip* 26:3825 (2026). PMCID PMC13208450
14. Chakrabarty et al. Cell-squeezing device with deep-learning readout. *Adv Healthc Mater* (2025).
    doi:10.1002/adhm.202502472

### Validation benchmarks (blood flow in micro-channels)

15. Wu W-T, Yang F, Antaki JF, Aubry N, Massoudi M. Study of blood flow in several benchmark micro-channels
    using a two-fluid approach. *Int J Eng Sci* 95:49–59 (2015).
    doi:10.1016/j.ijengsci.2015.06.004 · PMCID PMC4521229 · PMID 26240438 (`verified-primary`, full text read)
16. Zhao R, Marhefka JN, Shu F, Hund SJ, Kameneva MV, Antaki JF. *Ann Biomed Eng* 36:1130–1141 (2008).
    doi:10.1007/s10439-008-9494-z · PMID 18418710 · PMCID PMC6390961 (`verified-primary` abstract)
17. Sezer C, Kaya K, Tabatabaei Malazi M, Dalkılıç AS. Euler–Euler blood in a sudden expansion.
    *Micromachines* 17(3):316 (2026). doi:10.3390/mi17030316 · PMCID PMC13029181 (`verified`, full text read)
18. Gracka K, Lima R, Miranda J, Student V, Melka J, Ostrowski J. CFL validation in hyperbolic
    contractions, Euler–Euler vs Euler–Lagrange. *Comput Methods Programs Biomed* 226:107117 (2022).
    doi:10.1016/j.cmpb.2022.107117 (`verified`, CC BY, full text read)
19. Cell-free-layer CFD validation in a hyperbolic contraction (2022). PMID 36122496
    (`abstract` — paywalled)
20. Euler–Euler blood in oxygenators (2024). PMID 39531093
21. Core–plasma two-phase model (2025). PMCID PMC12758828
22. Wu W-T, Aubry N, Antaki JF, Massoudi M. *J Comput Phys* 376:112856 (2018). PMCID PMC8545272
    (full text read; source of the Zhao U/Re pairs, `secondary`)

### Rheology, haematocrit ranges, constants

23. Cho WS, Kensey RC. *Biorheology* 28(3–4):241–262 (1991) — Carreau fit μ₀ = 0.056 Pa·s,
    μ∞ = 0.00345 Pa·s, λ = 3.313 s, n = 0.3568, a = 2 (`secondary`: quoted via arXiv:2502.21168 and
    Kumar et al. 2018, *Comp Biol Med* 92:168–177; original paywalled, no identifier in `knowledge/`)
24. Giersiepen M, Wurzinger LJ, Opitz R, Reul H. *Int J Artif Organs* 13:300–306 (1990) — HI constants
    C = 3.62e-5, α = 2.416, β = 0.785 (`secondary`, via PMC10130582, PMC13291019; original paywalled)
25. Phillips RJ, Armstrong RC, Brown RA, Graham AL, Abbott JR. *Phys Fluids A* 4(1):30–40 (1992) —
    migration flux, K_c ≈ 0.41, K_μ ≈ 0.62 (`secondary`; original not re-read)
26. Garon A, Farinas M-I. *Artif Organs* 28(11):1016–1025 (2004). doi:10.1111/j.1525-1594.2004.00026.x
    (`secondary`)
27. Mimouni. Open J Biophys 6:29–33 (2016) — Quemada coefficients k₀ = 4.33, k∞ = 2.07, γ_c = 1.88 s⁻¹
    (`verified` as printed). doi:10.4236/ojbiphy.2016.62004
28. Northrop-Clewes CA, Thurnham DI. *J Blood Med* 4:11–22 (2013) — clinical Hct ranges.
    PMCID PMC3610441 (`verified`)
29. Waggoner. *J Adv Pract Oncol* 14(5) (2023) — 2016 WHO polycythaemia criteria.
    PMCID PMC10414534 (`verified`)
30. *Ther Clin Risk Manag* (2021). doi:10.2147/TCRM.S213020 · PMCID PMC7981161 (`verified`)
31. Rosenson RS, McCormick JB, Uretz O. *Clin Chem* 42:1189–1195 (1996) — plasma/blood viscosity
    (`abstract` only).
32. Kestin J, Sokolov V, Wakeham WA. *J Phys Chem Ref Data* 7(3):941 (1978) — water viscosity (`verified`).
33. Kang. *Biosensors* 16(4):216 (2026) — Krieger–Dougherty check fit (no DOI recorded in `knowledge/`).
34. Blood rheology review. arXiv:2609.20855 (2026).
35. Rheology in the aorta. PMCID PMC10669506.
36. RBC aggregation and viscosity. *PLoS ONE*. doi:10.1371/journal.pone.0199911
37. Per-Hct Carreau fitting method (preprint, no numbers printed). arXiv:2606.19870

### Cell-free layer, migration, cell mechanics

38. Fedosov DA, Caswell KK, Popel AS, Karniadakis GE. *Microcirculation* 17:615 (2010).
    doi:10.1111/j.1549-8719.2010.00056.x · PMCID PMC3529161 (`verified`, full text read)
39. Abkarian M, Viallat P. *Biophys J* 89:1055 (2005). PMCID PMC1366591 (`verified`, full text read)
40. Abkarian M, Lartigue C, Viallat P. *PRL* 88:068103 (2002). doi:10.1103/physrevlett.88.068103 (metadata)
41. Kumar A, Graham MD. *PRL* 109:108102 (2012). doi:10.1103/physrevlett.109.108102 (metadata)
42. Hariprasad J, Secomb TW. *Phys Rev E* 92:033008 (2015). PMCID PMC4765958 (`verified`, full text read)
43. Kim O, Kong H, Popel AS, Intaglietta M, Johnson PC. *Am J Physiol Heart Circ Physiol* (2007).
    doi:10.1152/ajpheart.01090.2006 (metadata)

### Surrogates, neural operators, multi-fidelity ML

44. COMSOL application paper 136422 (2024): DNN surrogate for blood-damage modelling in the FDA hemolysis
    benchmark. comsol.com/paper/deep-neural-network-surrogate-model-for-blood-damage-modeling-in-fda-
    hemolysis-benchmark-136422 (no DOI recorded in `knowledge/`)
45. Neural operator for platelet deformation in shear (2025). arXiv:2503.12074
46. Physics-guided surrogate for DLD design (2025). arXiv:2512.05649
47. Multi-scale RBC trauma in high-shear flows (2024). arXiv:2411.13002
48. HemoCell: open-source cell-resolved blood flow (LBM + immersed boundary). arXiv:2305.02752
49. Multi-fidelity FNO with transfer learning. arXiv:2304.06972
50. Multi-fidelity neural operators. arXiv:2608.04708
51. Multi-fidelity scaling laws. arXiv:2511.01830

### Own prior work (2023)

52. Nalla SK, Gereige E. *Shear encapsulation of RBCs for drug delivery* — MEC/BIO 586 report (2023-01-24)
    and 36-slide deck. Internal documents: `knowledge/papers/pdf/nalla-gereige-2023-report.pdf`,
    `knowledge/papers/pdf/nalla-gereige-2023-slides.pptx` (no DOI; owner-confirmed shareable, credit Gereige).

### Software documentation (equations read, never copied; cited as documentation)

53. COMSOL 6.3 CFD Module User's Guide: "The Euler–Euler Model Equations" (cfd_ug_fluidflow_multi.09.158),
    "Some Notes on Boundary Conditions" (09.159), "Pseudo Time Stepping for Laminar Flow Models"
    (cfd_ug_fluidflow_single.06.100); Reference Manual: stationary solver (comsol_ref_solver.36.118),
    stabilization (comsol_ref_modeling.19.090). URLs as recorded in [§2-theory](2-theory.md).
    All equations in `docs/` are written in our own notation.
54. OpenFOAM ESI v2306 source (lift/drag models, `multiphaseEulerFoam`, `reactingTwoPhaseEulerFoam`),
    verified via api.openfoam.com/2306 — see [§2-theory](2-theory.md) §6.
