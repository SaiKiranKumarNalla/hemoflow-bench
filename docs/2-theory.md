# 2 · Theory and mathematical models

All model equations in nondimensional (star) units.

## Contents

- [Mathematical model (all in star units)](#mathematical-model-all-in-star-units)
- [Nondimensionalization + multiphase math model (tutor mode)](#nondimensionalization--multiphase-math-model-tutor-mode)
- [Mathematical model (authoritative; supersedes the equation parts of 10)](#mathematical-model-authoritative-supersedes-the-equation-parts-of-10)
- [Blood models](#blood-models)
- [Haematocrit ↔ rheology: clinical ranges, φ-dependent whole-blood viscosity, carrier fluids](#haematocrit--rheology-clinical-ranges--dependent-whole-blood-viscosity-carrier-fluids)
- [Wu et al. (2015) Two-Fluid Blood Model Closures & Comparison with 2023 COMSOL Settings](#wu-et-al-2015-two-fluid-blood-model-closures--comparison-with-2023-comsol-settings)
- [Steady 2D two-fluid (Euler–Euler) formulation for the FEniCSx solver (V3 / V4)](#steady-2d-two-fluid-eulereuler-formulation-for-the-fenicsx-solver-v3--v4)
- [Cell-free layer (CFL) modelling — literature read and V4 closure proposal](#cell-free-layer-cfl-modelling--literature-read-and-v4-closure-proposal)

---

## Mathematical model (all in star units)


Authoritative source: [§2-theory](2-theory.md) (plus [§2-theory](2-theory.md), `17`, `18`, `19`). Every constant
carries its label: `verified` = original full text read; `secondary` = stated by a paper we read, citing
another we did not; `recalled` / `TODO: source` = not yet checked against an original. Nothing here is
invented; open items are marked.

### 1 · Scales and dimensionless groups

Scales: length $h_1$ (inlet height), velocity $U \equiv U_{\mathrm{mean}} = Q/A$ (a syringe pump sets $Q$;
owner decision 2026-11-24), time $h_1/U$, stress/pressure $\mu_{\mathrm{ref}} U/h_1$ with
$\mu_{\mathrm{ref}} = \mu_\infty$ (high-shear plateau), viscosity $\mu_\infty$, density $\rho$ (set to 1).
Starred quantities: $x^* = x/h_1$, $u^* = u/U$, $p^* = p h_1/(\mu_\infty U)$, $\dot\gamma^* = \dot\gamma h_1/U$,
$\tau^* = \tau h_1/(\mu_\infty U)$.

Physical scale of the reference device (2023 work): $h_1 = 50\,\mu\mathrm{m}$, $U = 0.25\,\mathrm{m/s}$,
$\rho = 1060\,\mathrm{kg/m^3}$, $\mu_\infty = 3.45\,\mathrm{mPa\cdot s}$ ⇒ $\tau_{\mathrm{ref}} \approx
17.25\,\mathrm{Pa}$, $\mathrm{Re} \approx 3.8$, $\mathrm{Cu} \approx 1.7\times10^4$. For a Carreau run the
consistent dimensional mapping is $h = \sqrt{\mathrm{Re}\,\mu_\infty \lambda/(\rho\,\mathrm{Cu})}$,
$U = \sqrt{\mathrm{Re}\,\mu_\infty\,\mathrm{Cu}/(\rho\lambda)}$, $\tau_{\mathrm{ref}} = \mu_\infty\mathrm{Cu}/\lambda$
(decision "consistent dimensional mapping", [9-decisions](9-decisions.md)).

| Group | Definition | Meaning | Reference value |
|---|---|---|---|
| $\mathrm{Re}$ | $\rho U h_1/\mu_\infty$ | inertia / viscous | 3.79–3.84 at the 2023 scale (plane channel, $H = 50\,\mu$m) |
| $\mathrm{Cu}$ | $\lambda U/h_1$ | fluid relaxation / flow time | $\approx 1.7\times10^4$ at the 2023 scale ($\lambda = 3.313$ s, `secondary`, Cho & Kensey 1991) |
| $\beta$ | $\mu_0/\mu_\infty$ | shear-thinning strength | $\approx 16.2$ for the Cho & Kensey fit (`secondary`) |
| $n$, $a$ | Carreau exponents | shape of the thinning curve | $n = 0.3568$, $a = 2$ (`secondary`) |
| $\phi$ | haematocrit | RBC volume fraction (M2/M2b/M3) | 0.20–0.50 whole blood; devices run 0.01–0.10 (`verified`, Bernardelli 2025 Table 2 / Table 3) |
| $a/h_1$ | RBC radius / channel | confinement | $\approx 0.056$ at $h_1 = 50\,\mu$m, MCV 90 fL |
| $\mathrm{Ca}_{cell}$ | $\mu_p \dot\gamma a / G_s$ | viscous stress / membrane shear modulus | $G_s$ `TODO: source` |
| $\mathrm{Pe}$ | $U h_1/D_c$ | advection / drug diffusion | $D_c$ (FITC-dextran 40 kDa) `TODO: source` |

Note: a VOF/interface-tension formulation is **not** used — an RBC suspension has no sharp fluid–fluid
interface; deformability enters through $\mathrm{Ca}_{cell}$, not a surface tension
([§2-theory](2-theory.md) §3.3 correction).

### 2 · M1 — single-phase generalized-Newtonian flow (the legacy corpus model)

Steady, incompressible, laminar; the RBCs are part of a shear-thinning fluid at fixed $\phi$:

$$\nabla^* \cdot u^* = 0, \qquad
\mathrm{Re}\, (u^* \cdot \nabla^*) u^* = -\nabla^* p^* + \nabla^* \cdot \left[ 2\,\mu_r(\dot\gamma^*)\, D^* \right],$$

with $D^* = \tfrac12(\nabla^* u^* + \nabla^* u^{*T})$, $\dot\gamma^* = \sqrt{2\,D^*:D^*}$ and the
Carreau–Yasuda ratio

$$\mu_r(\dot\gamma^*) = 1 + (\beta - 1)\,\left[1 + (\mathrm{Cu}\,\dot\gamma^*)^{a}\right]^{(n-1)/a}.$$

(The stored corpus pressure is viscous-scaled, $p^* = \mathrm{Re}\,p_{OF}$; the balance above is the correct
star form — the $1/\mathrm{Re}$ multiplies the whole viscous+pressure gradient, see the T9 flag in
(internal planning notes).)

- Viscosity models available: Newtonian ($\mu_r = 1$); Carreau–Yasuda (above; dimensional Cho & Kensey 1991
  fit: $\mu_0 = 0.056$ Pa·s, $\mu_\infty = 0.00345$ Pa·s, $\lambda = 3.313$ s, $n = 0.3568$, $a = 2$ —
  `secondary`); Casson and Quemada (Hct-aware) are specified in [§2-theory](2-theory.md) §2.2 with `TODO: source`
  for the fits.
- Boundary conditions: inlet fully developed (parabolic $u^* = 6y^*(1-y^*)$ for Newtonian; uniform
  $u^* = 1$ for Carreau, as generated); walls no-slip; outlet $p^* = 0$, zero gradient; 2D `empty` planes.
- Analytic reference (validation): plane Poiseuille $u^*(y^*) = 6 y^*(1-y^*)$ on $y^* \in [0,1]$,
  $\tau^*_{wall} = 6/Re$ (Newtonian), $u^*_{max} = 1.5$; in dimensional terms $\tau_{wall} = 6\mu U/H$
  (≈ 105 Pa at the 2023 scale; ≈ 140 Pa in a 50 µm tube).

### 3 · Post-processing metrics (same for all models; window and dose are computed per run)

**Shear-stress field:** $\tau^* = \mu_r(\dot\gamma^*)\,\dot\gamma^*$ from the solved field.

**Shear windows (screening statistics).** The 2023 window $[\tau_{lo}, \tau_{hi}] = [30, 60]$ Pa maps to
star units per run as $\tau^*_{lo,hi} = [30,60]\,h_1/(\mu_\infty U)$ (≈ [1.74, 3.48] at the 2023 scale).
Candidate variants from the assumption register ([§4-validation](4-validation.md)): W1 = 30–60 Pa, W2 = 15–45 Pa (Amar 2026
thresholds, `secondary`), W3 = 1–7 Pa physiological wall shear (negative control). The flux-weighted window
fraction is

$$f_{window} = \frac{\int_{\tau^*_{lo} \le \tau^* \le \tau^*_{hi}} \phi\, u_n\, dA}{\int \phi\, u_n\, dA}$$

(φ uniform cancels in M1 → fraction of the flow rate; in multiphase models weight by the RBC flux φ·w).

**Time-aware dose (decision criterion).** Pore opening needs *time* above a threshold, so one-sided gates
are also computed along RBC paths: exposure $t^*_{pore}$ above $\tau^*_{lo}$ (15 or 30 Pa), loaded fraction
$f_{load}$, and excess dose $J^* = \int \max(0, \tau^* - \tau^*_{lo})\, dt^*$. A per-design effective upper
bound $\tau_{hi,eff}$ is derived from $HI(\tau, t_{exp}) = HI_{lim}$ instead of assuming a static cap
(`HI_{lim}` `TODO: source`).

**Lagrangian exposure:** along a streamline / particle path, $t_{exp} = \int ds/|u|$ over the part of the
path inside the window.

**Hemolysis index (dose × time).** Giersiepen power law (constants `secondary`, [§4-validation](4-validation.md) row 6):

$$HI(\%) = C\,\tau^{\alpha}\, t^{\beta}, \qquad C = 3.62\times10^{-5},\ \alpha = 2.416,\ \beta = 0.785$$

($C = 3.62\times10^{-7}$ if HI is a fraction; τ in Pa, t in s; fit range τ = 120–535 Pa, t = 14–700 ms —
an extrapolation flag for our 0.18 s transits). Eulerian form (Garon & Farinas 2004, `secondary`): transport
the linearised damage $D_I = HI^{1/\beta}$,

$$u \cdot \nabla D_I = (C\,\tau^{\alpha})^{1/\beta}, \qquad
HI = C\, (\mu_c U/h_1)^{\alpha} (h_1/U)^{\beta}\,(D_I^*)^\beta .$$

Sanity checks (from [§4-validation](4-validation.md)): Bernardelli's 49.7 Pa × 0.18 s gives HI ≈ 0.12 %; 45 Pa reaches 0.5 %
only after ≈ 1.5 s ($t \propto \tau^{-3.08}$); at fixed geometry $HI \propto U^{\alpha-\beta} = U^{1.63}$
(Newtonian) — a falsifiable corpus check.

**Encapsulation index (2023 Eq. 1, with caveats).** As defined by Piergiovanni et al. 2020 (Eq. 1), transcribed
in the Nalla & Gereige 2023 report (p. 3):

$$I_{encap} = \frac{v \cdot \tau \cdot \phi}{v_{max} \cdot \tau_{max} \cdot \phi_{max}}.$$

Caveats ([§2-theory](2-theory.md) §4.5, [9-decisions](9-decisions.md)): (i) whether $v = |u|$; (ii) whether the maxima are per-case
or fixed references — if per-case, $I$ ranks only within one case and cannot compare geometries, so the
surrogate targets are the τ-statistics above unless fixed reference maxima are stated. Provisional decision:
maxima taken on the developed inlet channel, where the Poiseuille peak is exactly $2/(3\sqrt3) \approx 0.385$,
matching the 2023 range $0.2 < I < 0.4$. `TODO: verify against Piergiovanni 2020`.

### 4 · M2 / M2b — continuum multiphase (RBCs as a volume fraction)

**Mixture model (one momentum equation + φ transport):** mixture velocity $u_m$,

$$\nabla \cdot u_m = 0, \qquad
\rho_m (u_m \cdot \nabla) u_m = -\nabla p + \nabla \cdot [2\mu_m(\phi,\dot\gamma) D_m], \qquad
\partial_t \phi + u_m \cdot \nabla \phi = -\nabla \cdot N,$$

with the Phillips et al. 1992 shear-induced migration flux (`secondary`, constants `TODO: verify`)

$$N = -K_c a^2 \phi \nabla(\dot\gamma \phi) - K_\mu a^2 \phi^2 \dot\gamma \nabla(\ln \mu_m),
\qquad K_c \approx 0.41,\ K_\mu \approx 0.62,$$

producing the cell-free layer near the wall. Limitation: Phillips is rigid-sphere, collision-driven; RBC wall
lift is deformability-driven, so M2 likely **underpredicts** CFL thickness — a wall-lift term is needed
(§5). In star units the transport equation carries the small factor $(a/h_1)^2 \approx 3.1\times10^{-3}$, giving
a development length $L_{dev}/h_1 \approx 70 \gg L/h_1 \le 20$: a uniform inlet φ gives almost no CFL in a
short channel, so a pre-developed inlet φ(y) is used.

**Euler–Euler two-fluid (M2b).** Structure as documented for the COMSOL 6.3 "Euler–Euler Model, Laminar Flow"
interface (software documentation, cited; written here in our own notation, star units, phases plasma $c$ /
dispersed RBC $d$, $\phi = \phi_d$): per-phase continuity $\nabla \cdot (\phi_k u_k) = 0$ (incompressible), a
mixture continuity for the shared pressure, and per-phase momentum in the divided (non-conservative) form

$$\rho_k \left[\partial_t u_k + (u_k \cdot \nabla) u_k\right] = -\nabla p + \nabla \cdot \tau_k
+ F_{m,k}/\phi_{k,pos} + F_k,$$

with viscous stress $\tau_k = \mu_k^m (\nabla u_k + \nabla u_k^T)$ and interphase drag $F_{drag} = \beta (u_d - u_c)$
with, e.g., Schiller–Naumann $C_D = 24/Re_p (1 + 0.15 Re_p^{0.687})$ ⇒ $\beta = 18\mu_c \phi_d/d^2$ at low
$Re_p$ (Stokes limit), or Gidaspow (Wen–Yu/Ergun); mixture viscosity options include Krieger
$\mu_{mix} = \mu_c (1 - \phi_d/\phi_{max})^{-2.5\phi_{max}}$. Wu et al. 2015 (`verified-primary`, PMC4521229)
instead use per-phase stresses with a Yeleswarapu-type shear-thinning RBC viscosity
$\mu_{rbc}(\phi, \dot\gamma_{rbc})$ (their Eqs. 7, 9a–9c), Stokes drag with Rusche–Issa hindrance
$f(\phi) = e^{2.68\phi} + \phi^{0.43}$, and **Saffman shear lift** as the only interaction terms:

$$f_I = \frac{9\mu_f}{2a^2} f(\phi)\,(v_{rbc} - v_f)
+ \frac{3(6.46)(\rho_f\mu_f)^{1/2}}{4\pi a}\,\phi\,(2\,\mathrm{tr}\,D_f^2)^{-1/4}\, D_f (v_{rbc} - v_f).$$

Wu phase properties (`verified-primary`): $\mu_f = 0.96$ cP, $\rho_f = 1027$, $\rho_{rbc} = 1093$ kg/m³,
RBC diameter $a = 8\,\mu$m. Stock OpenFOAM ESI v2306 `multiphaseEulerFoam` has no lift and no φ-dependent
viscosity; `reactingTwoPhaseEulerFoam` adds lift models whose low-$Re_p$ limit (`LegendreMagnaudet`) reduces
analytically to the Saffman $\dot\gamma^{1/2}$ scaling with prefactor $6\times2.255/\pi^2 = 1.3709$
(between Wu's printed 0.7711 and rigid-sphere Saffman 3.0844) — verified in the v2306 source
([§2-theory](2-theory.md) §6).

### 5 · Deformability-driven wall lift (what sustains the CFL)

Steady two-fluid solutions with drag + slip-based Saffman lift converge with only ~1 % wall depletion — the
CFL is **not sustained** by slip lift, because RBCs barely slip. The owner's 2023 COMSOL time-dependent runs
showed the same: the CFL was visible in early snapshots and disappeared later ([§6-comsol-2023](6-comsol-2023.md)). Physical
chips keep a CFL (Gracka et al. 2022: 5 / 35 µm at Ht 5 %, `verified`; Bernardelli 2025 discussion). The
mechanism is **deformability-induced lateral migration**: Abkarian & Viallat 2005 (Biophys J 89:1055,
PMC1366591, read in full → `verified`) measured a viscous lift on tank-treading vesicles near a wall,

$$F_l = \eta\,\dot\gamma\, R^3 f(\nu)/h$$

(lift grows linearly with wall shear rate; $h$ = membrane–wall gap, $\nu$ = reduced volume, $f \approx 2$ for
RBCs at $\nu = 0.7$ per the authors' extrapolation; 31–155 pN at wall stress 0.2–1 Pa for R = 3 µm,
h = 350 nm). Per unit volume on the RBC phase this gives the V4 wall-lift closure
$f_{wl} = C_A \phi \mu_c \dot\gamma / y_{gap}\,\mathbf{n}_w$ with $C_A = 3f(\nu)/(4\pi) \approx 0.48$
([§2-theory](2-theory.md)). Combined with the Phillips counter-flux, the 1D steady wall balance predicts a CFL
thickness $\delta = (C_L/K_c)\, a/\phi_\infty$ — independent of flow rate (matches Sezer 2026 §3.2 and
Bernardelli's flat loading 30→50 µL/min) and $\propto 1/\phi$ (wider at low Ht: matches Fedosov 2010,
`verified`, and Zhao/Gracka). Validation targets: Gracka A/B (5 / 35 µm), Fedosov trends, Zhao depletion;
gate = the CFL persists at steady state.

### 6 · Pseudo-time stationary strategy

The Euler–Euler solvers exist as time-marching (PIMPLE) codes; steady targets are reached by marching in
pseudo-time. COMSOL's documented recipe (software documentation, [§2-theory](2-theory.md) §5): add
$\rho(u - u_{nojac})/\Delta\tilde t$ per nonlinear iteration with a local $\Delta\tilde t = CFL_{loc}\,h/|u|$
and the CFL schedule $CFL_{loc}(k) = 1.3^{\min(k,9)} + [k>20]\,9\,1.3^{\min(k-20,9)} +
[k>40]\,90\,1.3^{\min(k-40,9)}$, plus damped Newton. Our production variant is an own **steady** FEniCSx
two-fluid solver (Taylor–Hood P2/P1, monolithic Newton with continuation in φ and Re) — [§2-theory](2-theory.md);
OpenFOAM Euler–Euler is used for gates and cross-checks. Steady-state stop rule for the time-marching runs:
relative change of $\Delta p^*$, $x_r$, near-wall φ and CFL thickness < 1e-4 over the last 10 % of iterations
(decision 2026-11-24, [9-decisions](9-decisions.md)).

---

## Nondimensionalization + multiphase math model (tutor mode)


> **Superseded 2026-11-24 by [`11-math-model.md`](11-math-model.md).** The scales (§1) still hold; the VOF/interFoam
> multiphase model and the Ca ≈ 88 interpretation below are wrong for blood — see 11 §3.3.

Goal: write the governing equations **without units**, so one simulation covers a whole family of
physical cases (this is what makes a parameterized surrogate meaningful — same trick as the
parameterized PINN over aspect ratio h in `pinn/`).

### 1. Choose scales (the tutor step most people skip)

Pick scales from the *problem*, not from convenience:

| Quantity | Scale | Why |
|---|---|---|
| length | `H` = channel height (50 µm straight / 100 µm after expansion) | the geometry that sets shear |
| velocity | `Umax` = peak inlet velocity (2023 case: 0.25 m/s) | sets the shear rate γ̇ ~ Umax/H |
| time | `H / Umax` | advective time |
| pressure & stress | `μ Umax / H` (viscous scale) | balances the viscous term |
| viscosity | `μ` = plasma viscosity (~1.2 mPa·s; 2023 used 3.5 mPa·s whole-blood) | — |

Dimensionless variables: `x* = x/H`, `u* = u/Umax`, `t* = t Umax/H`, `p* = p H/(μ Umax)`,
`γ̇* = γ̇ H/Umax`.

### 2. Dimensionless numbers for the 2023 case (compute once, reuse forever)

With H = 50 µm, Umax = 0.25 m/s, μ = 3.5 mPa·s, ρ = 1060 kg/m³, σ ≈ 1e-5 N/m (RBC membrane):

- **Re** = ρ Umax H / μ = 1060·0.25·50e-6 / 3.5e-3 ≈ **3.8** → laminar, inertia small but present (matches the V3 `Re_in = 3.75` in the old FEM validation). This is why Stokes/PINN worked and why OpenFOAM laminar is safe.
- **Ca** (capillary) = μ Umax / σ ≈ 3.5e-3·0.25/1e-5 ≈ **88** → interface barely deforms; cells behave nearly rigid at this scale. Fine for MVP; note it in limitations.
- **We** = ρ Umax² H / σ = Re·Ca ≈ 330.
- **Carreau/Quemada relaxation**: λ* = λ Umax/H (dimensionless relaxation). For blood λ ~ 1–10 s → λ* ~ 5–50: strongly non-Newtonian in the high-shear zone near walls, Newtonian in the core.
- **Hematocrit φ** enters through the Quemada/Carreau relative viscosity μ_r(γ̇*, φ) — already dimensionless.

### 3. Dimensionless governing equations

Single phase (momentum, with viscous pressure scaling):

```
Re (∂u*/∂t* + u*·∇*u*) = -∇*p* + ∇*·(μ_r(γ̇*,φ) (∇*u* + ∇*u*ᵀ)) + (1/We) κ* ∇*α      (only in multiphase)
∇*·u* = 0
```

At Re ≈ 3.8 the left side is small → quasi-Stokes; OpenFOAM still solves the full NS, which is fine.

Multiphase — **two-phase incompressible VOF** (OpenFOAM `interFoam`), phase fraction α ∈ [0,1]
(α=1 cell/RBC phase, α=0 plasma):

```
∂α/∂t* + ∇*·(α u*) = 0                      (advection of the interface; MULES keeps α bounded)
ρ(α) = α ρ_cell + (1-α) ρ_plasma               (one-fluid density/viscosity mixing)
μ(α)  = α μ_cell + (1-α) μ_plasma(γ̇*,φ)
momentum: Re(...) = -∇*p* + ∇*·(μ(α)(∇*u*+∇*u*ᵀ)) + (1/We) κ* ∇*α
```

Surface tension enters as `σ κ ∇α` with κ = -∇·(∇α/|∇α|) the interface curvature; in viscous
scaling it is multiplied by **Ca**, in inertial scaling by **1/We** — same physics, two gauges.

Drug loading (the encapsulation index I): advection–diffusion of a scalar c on the cell phase with a
shear-gated source, e.g. `∂c/∂t* + ∇*·(c u*) = Pe⁻¹ ∇*²c + S(γ̇*, α)` — added in P2 once the flow
field is validated.

### 4. What this buys you

- One OpenFOAM case in `*`-units covers all geometrically similar channels — the campaign is over
  **dimensionless parameters** (h = H/L, expansion ratio, φ, λ*, Ca), not physical units.
- The parameterized PINN already speaks the same language (h is its parameter).
- Validation targets stay physical: reattachment 224 µm at Re_in = 3.75 (2023 COMSOL), max
  encapsulation at 0.56R — compare in `*`-units by multiplying back by H.

### 5. Numerical requirements (from the equations)

- Mesh: Δx* ≈ 1/50 (i.e. Δx = H/50) near walls where γ̇ is largest; graded away from walls.
- Time step (VOF/MULES): CFL = Umax Δt/Δx ≤ 0.5 → Δt* = 0.5 Δx*.
- `interFoam` needs `fvSchemes` with `div(phi,alpha)` MULES + `interfaceCompression`; `fvSolution`
  with PIMPLE nOuterCorrectors ≥ 2.

---

## Mathematical model (authoritative; supersedes the equation parts of 10)


Status: **DRAFT v2, 2026-11-24 — reviewed by the team + the team + the team; fixes applied** (see (internal planning notes)).
Constant labels: `verified: <paper, Table/Eq/p.>` (checked against the original PDF in `knowledge/papers/pdf/`),
`secondary: <citing paper>`, or `TODO: verify` (recalled value, no original in the repo yet).
Rule: every constant has a source from `knowledge/` or is marked `TODO: verify` / `TODO: source`. Nothing invented.

Scope: Stage A (device), 2D first. Three model families, compared on the same geometries:

| ID | Model | RBCs treated as | Solver (OpenFOAM ESI unless noted) | When |
|---|---|---|---|---|
| **M1** | Single-phase non-Newtonian (generalized Newtonian) | part of a shear-thinning fluid | `simpleFoam` + `BirdCarreau` / `Casson` | Wed–Thu (Saturday plan) |
| **M2** | Continuum **multiphase**: mixture (Hct transport + migration) | a continuum volume fraction φ | `driftFluxFoam`-style mixture **or** custom φ-equation (see §5.3) | after Saturday |
| **M2b** | Continuum multiphase: Euler–Euler two-fluid | a second interpenetrating fluid | `multiphaseEulerFoam` | comparison, optional |
| **M3** | **Particulate** (not continuum) | discrete rigid spheres (M3a) or deformable cells (M3b) | `DPMFoam`/`MPPICFoam` (M3a); HemoCell LBM-IBM (M3b, L3) | comparison, later |

---

### 1. Geometry and notation

- **G1 straight channel**: height H, length L (L/H ≤ 20), wall at y = 0 and y = H.
- **G2 sudden expansion**: inlet height h₁ = 50 µm → outlet height h₂ = 100 µm (expansion ratio E = h₂/h₁ = 2), step at x = 0.

| Symbol | Meaning | Unit |
|---|---|---|
| **u** = (u, v) | velocity | m/s |
| p | pressure | Pa |
| ρ | density | kg/m³ |
| μ, ν = μ/ρ | dynamic / kinematic viscosity | Pa·s, m²/s |
| **D** = ½(∇**u** + ∇**u**ᵀ) | strain-rate tensor | 1/s |
| γ̇ = √(2 **D**:**D**) | shear rate (scalar) | 1/s |
| τ = μ(γ̇) γ̇ | scalar shear stress | Pa |
| φ | haematocrit (RBC volume fraction) | – |
| a | RBC equivalent radius, a = (3·MCV / 4π)^(1/3) | m |

Note: OpenFOAM `simpleFoam` works with **kinematic** p/ρ and ν. Always divide by ρ when entering viscosities.

---

### 2. M1 — single-phase non-Newtonian flow

#### 2.1 Governing equations (steady, incompressible, laminar)
```
∇·u = 0
ρ (u·∇)u = −∇p + ∇·[ 2 μ(γ̇) D ]
```
The fluid is **generalized Newtonian**: stress is still 2μD, but μ depends on the local shear rate γ̇.
No viscoelasticity, no Hct variation (φ fixed ≈ 0.45 inside the fit).

#### 2.2 Viscosity models
| Model | μ(γ̇) | Parameters | Source | OpenFOAM keyword |
|---|---|---|---|---|
| Newtonian (baseline) | μ = const | μ = 3.5 mPa·s (whole blood, used in 2023) | 2023 project (`01-background`) | `Newtonian` |
| **Carreau–Yasuda** | μ∞ + (μ₀ − μ∞)[1 + (λγ̇)^a]^((n−1)/a) | μ₀ = 0.056 Pa·s, μ∞ = 0.00345 Pa·s, λ = 3.313 s, n = 0.3568, a = 2 | Cho & Kensey 1991 — `TODO: verify` | `BirdCarreau` (k = λ, a) |
| Casson | √τ = √τ_y + √(μ_c γ̇) (regularized) | τ_y, μ_c | `TODO: source` | `Casson` |
| Quemada (Hct-aware) | μ_p (1 − ½ k(γ̇) φ)^−2 | k₀, k∞, γ_c (fitted vs φ) | `TODO: source the fits` | custom (codedFunctionObject / library) |

OpenFOAM `BirdCarreau` takes **kinematic** values. Dimensional (ρ = 1060 kg/m³): `nu0 = 5.283e-5`, `nuInf = 3.255e-6` m²/s,
`k = 3.313` s, `n = 0.3568`, `a = 2`. In star units (ρ* = 1, length H, velocity U): ν* = μ/(μ_ref · Re), i.e.
`nu0* = (μ₀/μ_ref)/Re`, `nuInf* = (μ∞/μ_ref)/Re`, `k* = Cu`.

Why Carreau first: it has two Newtonian plateaus (μ₀ at low shear, μ∞ at high shear), so it never blows up
at γ̇ → 0 (centre line) like a pure power law does. In our channels γ̇ near the wall is ~10⁴ s⁻¹ (§4.3),
so the wall region is essentially at μ∞; non-Newtonian effects show up in the **low-shear core**
and in the **recirculation zone** of G2.

#### 2.3 Boundary conditions (both geometries)
| Boundary | u | p |
|---|---|---|
| inlet | fully developed profile (G1 analytic) or uniform U_in | zeroGradient |
| wall | no-slip, u = 0 | zeroGradient |
| outlet | zeroGradient | fixed p = 0 |
| front/back (2D) | `empty` | `empty` |

#### 2.4 Analytic reference: plane Poiseuille (Newtonian) — validation of G1
Fully developed flow, u = u(y), v = 0, constant pressure gradient G = −dp/dx:
```
μ d²u/dy² = −G,  u(0) = u(H) = 0
⇒ u(y) = (G / 2μ) · y (H − y)
u_max = G H² / (8μ)  at y = H/2
U_mean = (2/3) u_max
τ_wall = G H / 2 = 6 μ U_mean / H
γ̇(y) = |du/dy| = (G/μ)|H/2 − y|   (0 at centre, max at the wall)
```
Validation target: CFD centre-line u at the outlet within 2 % of u_max ((internal planning notes), P1).
For Carreau there is no closed form; validate with the ODE solved by shooting (1D, Python) — `TODO: owner writes`.

---

### 3. Nondimensionalization (applies to M1–M3)

#### 3.1 Scales (from the problem)
| Quantity | Scale |
|---|---|
| length | H (G1) or h₁ (G2) |
| velocity | **U ≡ U_mean = Q/A** (decision 2026-11-24; a syringe pump sets Q). The 2023 "inlet velocity 0.25 m/s" is treated as the mean: Re 3.37 fits U_mean = 0.25 m/s with μ/ρ ≈ 3.7e-6 m²/s; if 0.25 m/s were u_max, Re ≈ 1.9. Not proven from the deck. |
| time | H/U |
| pressure, stress | μ_ref U / H (viscous scaling, correct at Re ~ 1) |
| viscosity | μ_ref = μ∞ (high-shear plateau) |

x* = x/H, u* = u/U, p* = pH/(μ_ref U), γ̇* = γ̇H/U, μ_r = μ/μ_ref.

#### 3.2 Dimensionless equations (M1)
```
∇*·u* = 0
Re (u*·∇*) u* = −∇*p* + ∇*·[ 2 μ_r(γ̇*) D* ]
μ_r(γ̇*) = μ∞/μ_ref + (μ₀ − μ∞)/μ_ref · [1 + (Cu γ̇*)^a]^((n−1)/a)
```

#### 3.3 Dimensionless groups
| Group | Definition | Meaning | 2023 value (H = 50 µm, U = 0.25 m/s, ρ = 1060, μ = 3.5 mPa·s) |
|---|---|---|---|
| **Re** | ρ U H / μ_ref | inertia / viscous | 3.79 with μ = 3.5 mPa·s (3.84 with μ_ref = μ∞), plane channel, based on H. The 2023 value 3.37 is for a **circular tube**, D = 50 µm, with the COMSOL mixture viscosity — not directly comparable |
| **Cu** (Carreau number) | λ U / H | fluid relaxation time / flow time | 3.313 · 0.25 / 50e-6 ≈ 1.7·10⁴ ⇒ strongly shear-thinned everywhere except γ̇ ≲ 1/λ ≈ 0.3 s⁻¹ |
| μ₀/μ∞ | viscosity ratio | strength of shear thinning | 0.056/0.00345 ≈ 16 |
| n, a | Carreau exponents | shape | 0.3568, 2 |
| **φ** | haematocrit | already dimensionless | 0.20–0.50 |
| a/H | confinement | cell size / channel | a ≈ 2.8 µm (MCV 90 fL) → a/H ≈ 0.056 |
| **Ca_cell** | μ_p γ̇ a / G_s | viscous stress / membrane elasticity (RBC deformation) | G_s = membrane shear modulus `TODO: source (literature ~ µN/m)` |
| **Pe** | U H / D_c | advection / diffusion of drug | D_c for FITC-dextran 40 kDa `TODO: source` |

**Correction to `10-nondimensionalization.md`:** it used a VOF surface tension σ ≈ 1e-5 N/m and concluded
"Ca ≈ 88 → interface barely deforms". That is reversed (large Ca = viscous forces dominate = *large* deformation),
and an RBC suspension has no sharp fluid–fluid interface, so VOF/`interFoam` and σ are **not** the right
multiphase model. RBC deformability is described by Ca_cell with the membrane shear modulus, not σ.
Multiphase is handled by the mixture/Euler–Euler models in §5.

What the groups tell you: at Re ≈ 4 inertia is small but non-zero (G2 recirculation exists because of it — at
Re → 0 the expansion would not separate). So the campaign parameters are **(Re, E, φ, Cu, L/H)**, not physical units.

---

### 4. Derived quantities (post-processing; same for all models)

#### 4.1 Shear stress field
τ(x) = μ(γ̇) γ̇ — computed from the solved field (OpenFOAM `wallShearStress` for walls; a `codedFunctionObject`
or Python post-processing for the interior).

#### 4.2 Exposure time along a streamline / RBC path
t_exp = ∫ ds / |u| over the part of the path where τ is inside the shear window [τ_lo, τ_hi] = [30, 60] Pa
(2023 project, `01-background`). Paths from streamline tracing (M1, M2) or particle tracks (M3).

#### 4.3 Order of magnitude check (G1, Newtonian)
Plane channel: τ_wall = 6 μ U_mean / H. With μ = 3.5 mPa·s, U_mean = 0.25 m/s, H = 50 µm: τ_wall ≈ 105 Pa,
γ̇_wall ≈ 3·10⁴ s⁻¹. Circular tube (the 2023 geometry): τ_wall = 8 μ U_mean / D ≈ 140 Pa for D = 50 µm.
Either way the wall exceeds the 30–60 Pa window.

In fully developed flow the momentum balance gives τ linear from 0 at the centre to τ_wall at the wall, **for any
generalized-Newtonian μ(γ̇)** (τ = G·|y − H/2| in the channel, τ = τ_wall·r/R in the tube). So the window sits at
|y − H/2|/(H/2) (or r/R) between 30/τ_wall and 60/τ_wall. In G1 the rheology changes τ(y) only through G at fixed Q.

The 2023 optimum at ≈ 0.56–0.6R is **not** evidence about the window or the velocity: with φ uniform, Eq. 1 gives
I ∝ v·τ ∝ (1 − r²)·r, which peaks at r = R/√3 ≈ 0.577R at any flow rate.

#### 4.4 Hemolysis (power law)
```
HI(%) = C · τ^α · t^β,   C = 3.62e-5, α = 2.416, β = 0.785   (Giersiepen et al. 1990, `TODO: verify`)
```
Accumulated along paths (Lagrangian) or, in Eulerian form, via a transported linear damage variable
D_I = HI^(1/β): u·∇D_I = (C τ^α)^(1/β) (Garon & Farinas 2004 form — `TODO: source/verify`).
Units: HI in **%**, τ in Pa, t in s. If HI is used as a fraction, C = 3.62e-7.

#### 4.5 Encapsulation index I
```
I_encap = (v · τ · φ) / (v_max · τ_max · φ_max)        (Eq. 1)
```
Source: Eq. 1 of Piergiovanni et al. 2020 as reproduced in `knowledge/papers/pdf/nalla-gereige-2023-report.pdf`, p. 3
(image). `TODO: verify against Piergiovanni 2020`: (i) is v = |u|? (ii) are the maxima taken per case (over that
cross-section/domain) or fixed reference values? If per case, I is relative within one case and **cannot rank
different geometries or flow rates** — the surrogate must then use fixed reference maxima (state them) or the
τ-statistics. Until resolved, the surrogate targets are τ-statistics (frac_window, mean t_exp, HI), not I.

---

### 5. M2 — continuum multiphase (RBCs as a volume fraction)

#### 5.1 Mixture model (one momentum equation + φ transport)
Mixture velocity **u**_m, RBC fraction φ, plasma fraction 1 − φ:
```
∇·u_m = 0                                              (both phases incompressible, equal density assumption)
ρ_m (u_m·∇)u_m = −∇p + ∇·[ 2 μ_m(φ, γ̇) D_m ]
∂φ/∂t + u_m·∇φ = −∇·N                                   (haematocrit transport)
```
μ_m(φ, γ̇): Quemada (φ-aware, §2.2) or Carreau with Hct scaling `TODO: source`.
ρ_m = φ ρ_RBC + (1 − φ) ρ_p — ρ_RBC, ρ_p `TODO: source` (only matters if density differs; neutral buoyancy assumed).

#### 5.2 Shear-induced migration flux (Phillips et al. 1992)
```
N = − K_c a² φ ∇(γ̇ φ) − K_μ a² φ² γ̇ ∇(ln μ_m)
K_c ≈ 0.41, K_μ ≈ 0.62   (`TODO: verify constants`, [§2-theory](2-theory.md))
```
Term 1: cells move from high to low collision frequency (γ̇φ) → away from the wall.
Term 2: cells move down viscosity gradients.
Result: **cell-free layer** near the wall + concentrated core — the physics that matters for encapsulation,
because the high-shear wall zone is depleted of cells.

Closure and boundary conditions:
- **Walls:** N·n = 0 (no RBC flux through the wall). **Inlet:** φ = φ_in(y) — state whether uniform or pre-developed.
- **Viscosity term:** Phillips used μ(φ) only. Here ∇ln μ_m = (∂ln μ_m/∂φ) ∇φ (φ-part only, γ̇-dependence excluded) —
  decision to log in [9-decisions](9-decisions.md).
- **Centreline:** γ̇ = 0 there, so the steady balance drives φ → φ_max on the centreline. Regularize
  γ̇_ε = √(γ̇² + ε²) (or use a nonlocal shear rate) and cap φ ≤ φ_max (`TODO: source φ_max for RBCs`).
- **Nondimensional form** (N* = N·H²/(U a²)): ∂φ/∂t* + u*·∇*φ = −(a/H)² ∇*·N*, with (a/H)² ≈ 3.1·10⁻³.
- **Development length:** D_eff ~ K_c a² γ̇ φ; migrating across H/2 at γ̇_wall = 6U/H gives
  L_dev/H ~ (H/a)² / (24 K_c φ) ≈ 70 (longer with core shear rates) ≫ L/H ≤ 20. A uniform inlet φ gives almost no
  CFL in G1 → use a pre-developed inlet φ(y) from the 1D fully developed balance N_y = 0, or a longer domain.
- **Limitation:** Phillips is rigid-sphere, collision-driven migration. RBC CFL is driven mainly by
  deformability-induced wall lift, which this flux lacks → M2 likely **underpredicts** CFL thickness. Check against
  Wu et al. 2015 / Zhao et al. 2008 CFL; add a wall-lift term if needed (`TODO: source`).

#### 5.3 OpenFOAM implementation decision (open — log in DECISIONS.md)
- `driftFluxFoam` solves a mixture with a relative (slip) velocity, but its built-in slip models are
  settling-type (gravity), **not** Phillips shear-induced migration. Options:
  (a) one-way coupled: solve M1 flow, then a custom steady φ-equation with the Phillips flux (simplest);
  (b) two-way: add N as a source via `fvOptions`/coded solver and update μ_m(φ) each iteration;
  (c) Euler–Euler (M2b) with `multiphaseEulerFoam` + a lift/turbulent-dispersion-like model for migration.
- Recommendation: start (a), check CFL thickness vs Wu et al. 2015, then (b).

#### 5.4 M2b — Euler–Euler two-fluid (optional comparison)
Two continuity + two momentum equations (plasma p, RBC c), coupled by interphase drag M_d:
```
∂(φ_k ρ_k)/∂t + ∇·(φ_k ρ_k u_k) = 0
φ_k ρ_k (∂u_k/∂t + u_k·∇u_k) = −φ_k ∇p + ∇·(φ_k τ_k) ± M_d ± M_lift
```
−φ_k∇p is the standard shared-pressure form (as in `multiphaseEulerFoam`); no separate interfacial-pressure term needed.
Used by Wu et al. 2015 (OpenFOAM benchmark) and the 2026 sudden-expansion EE study (`02-literature`).
Closures (drag law, RBC-phase viscosity) `TODO: source from those papers`.

---

### 6. M3 — particulate RBCs (not continuum) — comparison model

#### 6.1 M3a: Euler–Lagrange rigid spheres (`DPMFoam`)
Fluid = plasma (Newtonian μ_p ≈ 1.2 mPa·s, `04-blood-models`). Each RBC = sphere radius a, mass m:
```
m dv_i/dt = F_drag + F_lift + F_pressure + F_collision,   F_drag = 6π μ_p a (u − v_i) · f(Re_p, φ)
dx_i/dt = v_i
```
Limits (state them in the report): rigid, spherical, valid for **dilute** suspensions — run at low φ (≤ 0.1)
or treat as qualitative at φ = 0.45. Value: gives per-cell shear-dose histories (like flow cytometry) directly.
Mesh limit: `DPMFoam` is point-particle — the fluid cells must be several times the particle diameter
(d = 2a ≈ 5.6 µm), which leaves only a few cells across H = 50 µm. The flow is then under-resolved, so M3a is
**qualitative** at this scale. Near-wall lubrication is not modelled.

#### 6.2 M3b: deformable cell-resolved (HemoCell, LBM + immersed boundary) — L3
Membrane with shear modulus G_s and bending modulus k_b (`TODO: source`); resolves CFL formation from first principles.
Hours per run; used only to calibrate migration constants and shear thresholds (`05-model-hierarchy`, L3).

#### 6.3 Comparison protocol (M1 vs M2 vs M3)
Same geometry (G1 then G2), same flow rate, same φ where valid. Compare:
1. velocity profile u(y) at x = L/2 (relative L2 difference),
2. haematocrit profile φ(y) and **cell-free-layer thickness** (M2, M3 only),
3. flux-weighted fraction of RBCs in the window and exposure-time distribution:
   f_window = ∫_{30 ≤ τ ≤ 60 Pa} φ u dA / ∫ φ u dA   (M1: φ uniform cancels → fraction of the flow rate;
   M3: fraction of particles crossing the section inside the window),
4. hemolysis index,
5. cost (CPU-hours per case).
Hypothesis (may fail — see the §5.2 limitation): M1 overestimates cells at the wall → overestimates both
encapsulation and hemolysis; M2 corrects that cheaply; M3 confirms M2's CFL.

---

### 7. Surrogates in math form

#### 7.1 Parameterized PINN (extends `pinn/microchannel.py`)
M1 network u_θ(x*, y*; Re, Cu) → (u*, v*, p*). φ becomes an input only when μ_r = μ_r(γ̇*, φ) (Quemada, M2);
in M1, μ_r has no φ, so a φ input would be ignored by the physics loss. Loss:
```
L(θ) = w_pde ‖R_mom‖² + w_div ‖∇*·u*‖² + w_bc ‖u* − u*_bc‖²_∂Ω + w_data ‖u* − u*_CFD‖²_sparse
R_mom = Re (u*·∇*)u* + ∇*p* − ∇*·[2 μ_r(γ̇*) D*]
```
Evaluated at random collocation points; the parameters (Re, Cu) — plus φ for M2 — are network inputs so one
training covers a family.

#### 7.2 Data-driven surrogate (Friday)
Map 𝒢: (Re, E, φ, Cu, L/H) → { scalars: Δp*, x_r/h₁ (reattachment), frac_window, mean t_exp*, HI }
or → fields (u*, v*, τ*) on a fixed grid. Models: MLP/GP (scalars) → DeepONet/FNO (fields), metric: relative L2 on a held-out set.

---

### 8. Open items (blockers before data generation)
1. ~~Encapsulation index Eq. 1 — transcribe.~~ Transcribed (§4.5) from the 2023 report; still check v and the
   maxima definitions against Piergiovanni 2020.
2. ~~2023 inlet velocity: mean or max?~~ Decided: U ≡ U_mean = Q/A (§3.1). Log in [9-decisions](9-decisions.md).
3. Verify Carreau (Cho & Kensey 1991), Giersiepen 1990, Phillips 1992, Garon & Farinas 2004 against the originals
   (no original PDFs in the repo yet — values are recalled/secondary).
4. Quemada fits vs φ; RBC/plasma densities; G_s; drug diffusivity; φ_max for RBCs.
5. Decide M2 implementation route (§5.3) and log it.
6. G1 for comparison with 2023: plane channel vs axisymmetric wedge (2023 used a circular tube).
7. M2 CFL realism: Phillips lacks deformability wall lift (§5.2) — source and add a wall-lift term if needed.

---

## Blood models


### Patient blood vector (input)
| Symbol | Meaning | Typical range | Used for |
|---|---|---|---|
| Hct (φ) | Haematocrit | 0.20–0.50 (device suspensions may be lower) | Mixture volume fraction, viscosity |
| μp | Plasma viscosity | ~1.1–1.6 mPa·s | Plasma (Newtonian) phase |
| Alb | Albumin | `TODO: range + effect from 2025 follow-up` | Loading-rate correction |
| EI | RBC elongation index (ektacytometry) | `TODO: source` | Scales pore-opening and hemolysis thresholds |
| MCV | Mean cell volume | ~80–100 fL | RBC radius a in migration model |
| Fib | Fibrinogen (optional) | `TODO` | Aggregation / low-shear viscosity |

All patient vectors are **synthetic**, sampled from published ranges. No real patient data.

### Rheology (choose one per study; record in DECISIONS.md)
- **Carreau–Yasuda:** μ(γ̇) = μ∞ + (μ0 − μ∞)[1 + (λγ̇)^a]^((n−1)/a).
  Common whole-blood Carreau fit (a = 2): μ0 = 0.056 Pa·s, μ∞ = 0.00345 Pa·s, λ = 3.313 s, n = 0.3568
  (Cho & Kensey 1991 — `TODO: verify values against the original paper`). Fixed Hct ≈ 0.45, so it must be made
  Hct-dependent for patient inputs.
- **Quemada (Hct-aware, preferred for patient inputs):** μ = μp (1 − ½ k φ)^−2,
  k(γ̇) = (k0 + k∞ (γ̇/γc)^½) / (1 + (γ̇/γc)^½), with k0, k∞, γc fitted vs φ. `TODO: source the fits.`

### Haematocrit transport with shear-induced migration (Phillips et al. 1992)
∂φ/∂t + u·∇φ = −∇·N, with
N = −Kc a² φ ∇(γ̇ φ) − Kμ a² φ² γ̇ ∇(ln μ), Kc ≈ 0.41, Kμ ≈ 0.62 (`TODO: verify constants`).
Produces the cell-free layer and the tubular pinch ring.

### Damage and encapsulation
- **Hemolysis (power law):** HI(%) = C τ^α t^β (Giersiepen et al. 1990: C = 3.62e-5, α = 2.416, β = 0.785;
  `TODO: verify`). Accumulate along RBC paths (MC-1) or as a transported scalar.
- **Encapsulation index I:** Piergiovanni 2020, Eq. 1. `TODO: transcribe exactly; do not approximate.`
- **Shear window:** 30–60 Pa (2023 project, from literature) — scale with EI once calibrated (L3).

---

## Haematocrit ↔ rheology: clinical ranges, φ-dependent whole-blood viscosity, carrier fluids


Written 2026-11-24. Labels follow [§4-validation](4-validation.md):
`verified` = the original full text (or its PDF/HTML) was read and the number seen there; `secondary` = a paper we
read states it and cites a source we did not read; `abstract` = abstract only; `derived` = computed here from
`verified` inputs by code in this repo (the formula is ours, the inputs are sourced); `TODO` = no source yet.

Purpose: give the paired corpus (V1 `N`, V2 `GN`) a defensible μ_eff(φ) and a φ-dependent shear-thinning law, and
fix the carrier-fluid viscosity as a run parameter (CORPUS_PLAN §6.3, [§4-validation](4-validation.md) A4/A13/A14).

### 1. Clinical haematocrit φ (volume fraction, L/L)

| Quantity | Value | Label | Source (read) |
|---|---|---|---|
| Normal range, men | 0.40–0.50 | verified | Northrop-Clewes & Thurnham, *J Blood Med* 2013;4:11–22, §"Hematocrit or packed cell volume", PMC3610441 |
| Normal range, women | 0.36–0.46 | verified | same |
| Reference mean, men 18–45 y | 0.445 (WHO mean; mean − 2 SD = 0.39) | verified as quoted | same paragraph (WHO value quoted there → the WHO source itself is `secondary`) |
| Reference mean, women 18–45 y | 0.400 (mean − 2 SD = 0.34) | verified as quoted | same |
| Anaemia (operational) | φ below the sex-specific reference range (< 0.40 men, < 0.36 women); WHO −2 SD anchors 0.39 / 0.34 | verified ranges; "below range = anaemia" is that paper's framing | same |
| Polycythaemia (WHO 2016 PV major criterion) | φ > 0.49 (men), φ > 0.48 (women) | verified | Waggoner, *J Adv Pract Oncol* 2023;14(5), Table 1 "2016 WHO criteria", PMC10414534 |
| Same thresholds, independent statement | Hct 49 % men / 48 % women | verified | *Ther Clin Risk Manag* 2021, doi:10.2147/TCRM.S213020, PMC7981161 |
| PV treatment target | φ < 0.45 | verified | same (RESPONSE-2 endpoint) |
| Absolute erythrocytosis anchor values | 55.5 % (men), 49.5 % (women) | verified as printed | PMC10414534 Table 1 note b |

**Pinned clinical designs for the corpus (our choice, derived from the thresholds above):**
anaemic **φ = 0.30** (below both reference ranges), normal **φ = 0.445** (male reference mean), polycythaemic
**φ = 0.55** (above the 0.49/0.48 thresholds and equal to the round "55 %" used as an erythrocytosis anchor).
`TODO: source` for a *population mean* of anaemic / polycythaemic patients — no primary series was found.

### 2. φ-dependent whole-blood viscosity: the Quemada law

The form and coefficients below are printed in a peer-reviewed open-access paper that compares rheological models
of human blood; the Quemada originals (Quemada 1976/1977) are paywalled and were **not** read.

Formula (as printed, eq. 4 of the source), with H = haematocrit and γ̇ the shear rate:

    η(γ̇, H) = η_p · [ 1 − ½ · (k0 + k∞·√(γ̇/γ_c)) / (1 + √(γ̇/γ_c)) · H ]^(−2)

| Quantity | Value | Label | Source |
|---|---|---|---|
| k0 | 4.33 | verified | Mimouni, *Open J Biophys* 2016;6:29–33, §3.1.3, doi:10.4236/ojbiphy.2016.62004 |
| k∞ | 2.07 | verified | same |
| γ_c | 1.88 s⁻¹ | verified | same |
| η_p used in the fit | 1.2 × 10⁻³ Pa·s | verified | same |
| Hct of the fitted data | H = 0.45 (single healthy donor) | verified | same §3.2 |
| Temperature | **not stated in the paper** | — | **FLAG** |

Limits used by our V1/V2 parameters (derived from the verified coefficients):

- high shear (γ̇ ≫ γ_c): μ∞(φ) = μ_p (1 − ½ k∞ φ)^(−2)
- at finite reference shear γ̇_ref: μ_eff(φ, γ̇_ref) from the full law above

**Validity guard (found while implementing):** the bracket must stay positive, ½·k(γ̇)·φ < 1. With k0 = 4.33 the
*zero-shear* limit is singular above φ ≈ 0.46, so the law must be evaluated at a finite shear rate for
φ ∈ [0.5, 0.6] — at γ̇ ≥ 100 s⁻¹ the bracket is positive up to φ = 0.6 (checked numerically). Code raises rather
than returning a negative viscosity.

**Cross-check against measured blood (abstract only):** Rosenson, McCormick & Uretz, *Clin Chem* 1996;42:1189–1195
report, at 37 °C for healthy adults, 3.26 ± 0.43 mPa·s at 100 s⁻¹, 4.37 ± 0.60 at 50 s⁻¹, 5.46 ± 0.84 at 1 s⁻¹
(`abstract`). Evaluating the Quemada law at H = 0.45, μ_p = 1.2 mPa·s (`campaign/blood_rheology.py`, T7):

| γ̇ (s⁻¹) | Quemada model (mPa·s) | measured (mPa·s) | model / measured |
|---|---|---|---|
| 100 | 5.37 | 3.26 | 1.65 |
| 50 | 5.88 | 4.37 | 1.35 |
| 1 | 20.80 | 5.46 | 3.81 |

So the shape is right (monotone thinning, right order of magnitude in the 50–100 s⁻¹ band that dominates these
devices) but the single-donor fit **overestimates** the population mean by 35–65 % at 50–100 s⁻¹ and by ~4× at
1 s⁻¹, where the Quemada form keeps thinning too slowly. Consequence for the corpus: μ_eff all levels scale with
this law, so V1/V2 absolute viscosities are ~1.4–1.7× high in the device band; the *ratio* μ_eff(φ)/μ_c, which is
what separates the haematocrit levels, is much less sensitive. **FLAG for the article; a Chien- or Brooks-based
φ-dependent Carreau fit would be preferred as soon as a readable source exists.**

Alternative φ-dependent law (for a robustness check, not used): Krieger–Dougherty μ = μ0(1 − φ/φ_m)^(−α φ_m) with
φ_m = 1.0, α φ_m = 1.3047, μ0 ≈ 1.19 (unit printed only as a number, apparently mPa·s), fitted to Pirofsky
haematocrit–viscosity data, R² = 0.9176 — Kang, *Biosensors* 2026;16(4):216, §3.1 and Fig. 2(A ii), verified as
printed. The ambiguous μ0 unit is why this is a check and not the corpus law.

### 3. φ-dependent Carreau(–Yasuda) fit — **GAP, and what we do instead**

**No readable publication prints μ0(φ), μ∞(φ), λ(φ), n(φ), a(φ) for whole blood.** What exists:
a 2026 preprint documents the *method* (per-Hct fits at 25/65/95 %, PCHIP interpolation in log space, 45 % held out)
but prints no numbers (arXiv:2606.19870, `preprint`), and a paywalled *J Biomech Eng* table titled "Carreau model
parameters with different hematocrits" was seen only as a caption and must not be used unread.

What *is* verified is Sezer 2026's mixture Carreau–Yasuda choice: Jung et al. kinetics with **λ = 0.110 s**,
m(α)/n(α) polynomials split at γ̇ = 6 s⁻¹ (Sezer et al., *Micromachines* 17(3):316, 2026, Eqs. 5–11, verified;
[§8-literature](8-literature.md) row 1). Their m(α), n(α) coefficient values are figures/equations we did not read out.

**Corpus construction (derived, FLAG):** V2 `GN` uses a BirdCarreau law whose *levels* come from the verified Quemada
law and whose *time constant* is the verified λ:

| Parameter | How |
|---|---|
| μ∞(φ) | Quemada high-shear limit μ_c (1 − ½ k∞ φ)^(−2) |
| λ | 0.110 s (Jung et al. via Sezer 2026) |
| a | 2 (Carreau–Yasuda exponent used throughout the corpus, `TODO: verify` — Cho & Kensey 1991 is the usual citation) |
| μ0(φ), n(φ) | least-squares fit of the BirdCarreau curve to the Quemada law over γ̇ ∈ [10, 10⁴] s⁻¹ at that φ |

Every record stores the fit residual (`carreau_fit_max_rel_err`), so a reader can see how well V2 approximates the
verified law. This is **our construction, not a published fit** — the article must say so.

### 4. Carrier-fluid viscosity μ_carrier (a run parameter, not a constant)

| Fluid | μ (mPa·s) | Label | Source |
|---|---|---|---|
| Plasma, as used in Wu 2015 and Sezer 2026 | **0.96** | verified | Wu 2015 §3 ([§2-theory](2-theory.md)); Sezer 2026 §2.2 ([§8-literature](8-literature.md)) |
| Plasma, ICSH selected-method range at 37 °C | 1.16–1.33 (1.50–1.72 at 25 °C) | secondary | ICSH 1984, *J Clin Pathol* 37:1147–1152, as quoted by ScienceDirect topic page (original blocked) |
| Plasma, healthy adults at 37 °C | 1.39 ± 0.08 (n = 126) | abstract | Rosenson et al. 1996, *Clin Chem* 42:1189–1195 |
| PBS / GASP (PBS + 1 % BSA) buffers | 1.12–1.68 (suspension values, Bernardelli Table 3) | verified | Bernardelli et al. 2025, PMC12006216 Table 3 ([§4-validation](4-validation.md) A14) |
| Water, 20 °C / 25 °C | 1.002 / 0.890 | verified | Kestin, Sokolov & Wakeham, *J Phys Chem Ref Data* 1978;7(3):941, Table 1 |
| PBS, PBS+1 % BSA | **no verified measured value** | `TODO: source` | searched, none readable |

Corpus rule (CORPUS_PLAN §6.3): **plasma μ_c = 0.96 mPa·s for φ > 0.1** (the Wu/Sezer convention, so the corpus stays
comparable to the validation cases), and a **buffer sampled in 1.1–1.7 mPa·s for φ ≤ 0.1** (device regime, Bernardelli
Table 3). μ_ref of a run is its own μ_carrier; the four variants of a design therefore share Re.

### 5. Open items (FLAG)

1. Temperature: the Quemada coefficients have no stated temperature (`verified` numbers, unknown T). The corpus is
   isothermal and non-dimensional, so this only affects the *dimensional* μ_eff used to build Re.
2. k0, k∞, γ_c are fitted at H = 0.45; using them at φ = 0.01–0.60 is an extrapolation of the *same* law, not a new
   fit. No Hct-dependent k0/k∞/γ_c set could be read out (Roman-type polynomial fits appear only as figures).
3. No published φ-dependent Carreau fit (§3) → V2 is our construction.
4. Buffer viscosity range is taken from measured *suspension* viscosities, not from a PBS/BSA rheology measurement.
5. Population means for anaemic/polycythaemic Hct are `TODO`.

---

## Wu et al. (2015) Two-Fluid Blood Model Closures & Comparison with 2023 COMSOL Settings


**Primary source (`verified-primary`):** Wu, W.-T., Yang, F., Antaki, J. F., Aubry, N., & Massoudi, M. (2015).
*Study of blood flow in several benchmark micro-channels using a two-fluid approach.*
*International Journal of Engineering Science*, 95, 49–59. DOI: `10.1016/j.ijengsci.2015.06.004`,
PMID `26240438`, PMCID `PMC4521229` (author manuscript NIHMS705281 read in full via NCBI BioC JSON and
PMC XML `efetch` on 2026-11-24).

**Rule:** Equations, constants, and boundary conditions below are transcribed verbatim from PMC4521229
(including manuscript notation and noted discrepancies/typos). Any quantity not stated in the manuscript is
explicitly marked `not found`.

---

### 1. Governing Equations (Wu 2015 §2.1 & §3)

Blood is modeled via mixture theory (Theory of Interacting Continua) as two interpenetrating incompressible
continua: continuous **plasma** (subscript $f$) and dispersed **red blood cells** (subscript $rbc$), with
RBC volume fraction (hematocrit) $\phi$ and plasma volume fraction $1 - \phi$. Effective (bulk) phase
densities satisfy volume additivity:

$$\rho_f = (1 - \phi)\rho_{f0}, \qquad \rho_{rbc} = \phi\,\rho_{rbc0}, \qquad \rho_{\text{mix}} = \rho_f + \rho_{rbc}$$

where $\rho_{f0}$ and $\rho_{rbc0}$ are the pure reference densities of plasma and RBCs. (In §3, Eqs. (12)–(13),
Wu 2015 reuses the symbols $\rho_f$ and $\rho_{rbc}$ for the pure densities; velocities are typeset as
$\boldsymbol{\nu}_f, \boldsymbol{\nu}_{rbc}$ in the PMC XML and written $\mathbf{v}_f, \mathbf{v}_{rbc}$ below.)

#### Conservation of mass (§2.1)

$$\frac{\partial \rho_f}{\partial t} + \operatorname{div}(\rho_f \mathbf{v}_f) = 0 \tag{1}$$

$$\frac{\partial \rho_{rbc}}{\partial t} + \operatorname{div}(\rho_{rbc} \mathbf{v}_{rbc}) = 0 \tag{2}$$

#### Balance of linear momentum (§2.1)

$$\rho_f \frac{D^f \mathbf{v}_f}{Dt} = \operatorname{div}(\mathbf{T}_f) + \rho_f \mathbf{b}_f + \mathbf{f}_I \tag{3}$$

$$\rho_{rbc} \frac{D^{rbc} \mathbf{v}_{rbc}}{Dt} = \operatorname{div}(\mathbf{T}_{rbc}) + \rho_{rbc} \mathbf{b}_{rbc} - \mathbf{f}_I \tag{4}$$

with component material derivatives defined inline in §2.1 for $\alpha \in \{f, rbc\}$:

$$\frac{D^\alpha \beta}{Dt} = \frac{\partial \beta}{\partial t} + \mathbf{v}_\alpha \cdot \operatorname{grad}\beta, \qquad
\frac{D^\alpha \mathbf{w}}{Dt} = \frac{\partial \mathbf{w}}{\partial t} + (\operatorname{grad}\mathbf{w})\,\mathbf{v}_\alpha$$

where $\mathbf{T}_f, \mathbf{T}_{rbc}$ are the partial Cauchy stress tensors, $\mathbf{b}_f, \mathbf{b}_{rbc}$
are body forces, and $\mathbf{f}_I$ is the momentum exchange (interaction force) acting on plasma from RBCs
(equal and opposite $-\mathbf{f}_I$ on RBCs). Angular momentum balance (without couple stresses) enforces
symmetry of the total Cauchy stress tensor $\mathbf{T}_m = \mathbf{T}_1 + \mathbf{T}_2$, where the manuscript
writes $\mathbf{T}_1 = (1-\phi)\mathbf{T}_f$, $\mathbf{T}_2 = \phi\mathbf{T}_{rbc}$ (noting a manuscript
typo: "reduces to that of a pure fluid as $\phi \to 1$" instead of $\phi \to 0$).

#### Expanded incompressible momentum equations (§3)

With general 3D fields $\mathbf{v}_f(x,y,z;t)$, $\mathbf{v}_{rbc}(x,y,z;t)$, $\phi(x,y,z;t)$ (Eq. 11) and
incompressibility ($\operatorname{tr}\mathbf{D}_f = \operatorname{tr}\mathbf{D}_{rbc} = 0$), substituting
Eqs. (5)–(10) into Eqs. (3)–(4) gives:

- **Plasma (§3, Eq. 12):**

$$(1-\phi)\rho_f \left[\frac{\partial \mathbf{v}_f}{\partial t} + (\operatorname{grad}\mathbf{v}_f)\mathbf{v}_f\right]
= -\operatorname{grad}\!\big((1-\phi)p\big)
  + \operatorname{div}\!\big(2\mu_f(1-\phi)\mathbf{D}_f\big)
  + \rho_f(1-\phi)\mathbf{b}_f
  + \frac{9\mu_f}{2a^2} f(\phi)(\mathbf{v}_{rbc} - \mathbf{v}_f)
  + \frac{3(6.46)(\rho_f \mu_f)^{1/2}}{4\pi a}\,\phi\,(2\operatorname{tr}\mathbf{D}_f^2)^{-1/4}\mathbf{D}_f(\mathbf{v}_{rbc} - \mathbf{v}_f) \tag{12}$$

- **RBCs (§3, Eq. 13):**

$$\phi\,\rho_{rbc} \left[\frac{\partial \mathbf{v}_{rbc}}{\partial t} + (\operatorname{grad}\mathbf{v}_{rbc})\mathbf{v}_{rbc}\right]
= -\operatorname{grad}(\phi p)
  + \operatorname{div}\!\left[\left(\mu_\infty(\phi) + \big(\mu_0(\phi) - \mu_\infty(\phi)\big)\frac{1 + \ln\!\big(1 + k(2\operatorname{tr}\mathbf{D}_{rbc}^2)^{1/2}\big)}{1 + k(2\operatorname{tr}\mathbf{D}_{rbc}^2)^{1/2}}\right)\mathbf{D}_{rbc}\right]
  + \rho_{rbc}\phi\,\mathbf{b}_{rbc}
  - \frac{9\mu_f}{2a^2} f(\phi)(\mathbf{v}_{rbc} - \mathbf{v}_f)
  - \frac{3(6.46)(\rho_f \mu_f)^{1/2}}{4\pi a}\,\phi\,(2\operatorname{tr}\mathbf{D}_f^2)^{-1/4}\mathbf{D}_f(\mathbf{v}_{rbc} - \mathbf{v}_f) \tag{13}$$

> **Pressure-gradient note:** Eqs. (12)–(13) retain $-\operatorname{grad}((1-\phi)p)$ and
> $-\operatorname{grad}(\phi p)$ directly from $\operatorname{div}(-p(1-\phi)\mathbf{I})$ and
> $\operatorname{div}(-p\phi\mathbf{I})$, which differs from the standard shared-pressure Euler–Euler form
> $-(1-\phi)\nabla p$ and $-\phi\nabla p$ (used in OpenFOAM `multiphaseEulerFoam` and [§2-theory](2-theory.md) §5.4)
> by the interfacial-pressure-like term $\pm p\nabla\phi$.

---

### 2. Stress & Viscosity Closures (Wu 2015 §2.2)

By Adkins' principle of phase separation, $\mathbf{T}_f$ depends only on plasma kinematics and $\mathbf{T}_{rbc}$
only on RBC kinematics.

#### Plasma stress tensor (Newtonian)

$$\mathbf{T}_f = \big[-p(1-\phi) + \lambda_f(1-\phi)\operatorname{tr}\mathbf{D}_f\big]\mathbf{I} + 2\mu_f(1-\phi)\mathbf{D}_f \tag{5}$$

where $\mathbf{D}_f = \frac{1}{2}[(\operatorname{grad}\mathbf{v}_f) + (\operatorname{grad}\mathbf{v}_f)^T]$
and $\lambda_f, \mu_f$ are the constant bulk/shear viscosities of pure plasma ($\lambda_f$ drops out under
incompressibility; its numeric value is `not found`).

#### RBC stress tensor & shear-thinning viscosity (generalized Oldroyd-B / Yeleswarapu et al.)

$$\mathbf{T}_{rbc} = \big[-p\phi + \beta_{20}(\phi + \phi^2)\operatorname{tr}\mathbf{D}_{rbc}\big]\mathbf{I} + \mu_{rbc}(\phi, \operatorname{tr}\mathbf{D}_{rbc})\,\phi\,\mathbf{D}_{rbc} \tag{6}$$

$$\mu_{rbc}(\phi, \operatorname{tr}\mathbf{D}_{rbc})\,\phi
= \mu_\infty(\phi) + \big(\mu_0(\phi) - \mu_\infty(\phi)\big)
  \frac{1 + \ln\!\big(1 + k(\phi)\,(2\operatorname{tr}\mathbf{D}_{rbc}^2)^{1/2}\big)}{1 + k(\phi)\,(2\operatorname{tr}\mathbf{D}_{rbc}^2)^{1/2}} \tag{7}$$

where $\dot{\gamma}_{rbc} = (2\operatorname{tr}\mathbf{D}_{rbc}^2)^{1/2}$ is the RBC-phase scalar shear rate.
Mixture viscosity is written as the volume-weighted sum:

$$\mu_{\text{mix}} = \mu_f(1-\phi) + \mu_{rbc}\phi \tag{8}$$

Calibrated to Brooks' experimental data for RBCs in ACD-plasma (Wu 2015 Fig. 1):

$$\mu_0(\phi) = 537.002\,\phi^3 + 55.006\,\phi^2 - 0.129\,\phi \tag{9a}$$

$$\mu_\infty(\phi) = 27.873\,\phi^3 - 21.218\,\phi^2 + 14.439\,\phi \tag{9b}$$

$$k = 11 \tag{9c}$$

**Transcription notes on Eqs. (6)–(9c):**
1. **Factor of 2:** Eqs. (6) and (13) omit the factor of $2$ before $\mu_{rbc}\phi\mathbf{D}_{rbc}$ (whereas
   Eq. (5) has $2\mu_f(1-\phi)\mathbf{D}_f$). Transcribed verbatim as printed in PMC4521229.
   **Project decision (2026-11-24, the team / T6):** keep `stress_factor_half=True` in `wu2015_mu_ratio`
   (i.e. $\mu_{rbc,\text{OF}} = \frac{1}{2}\mu_{rbc,\text{Wu}}$ so OpenFOAM's $2\alpha\mu\mathbf{D}$ reproduces
   Wu's printed $\mu_{rbc}\phi\mathbf{D}_{rbc}$ verbatim, keeping Wu's published results reproducible).
2. **Argument typo:** The left-hand side of Eqs. (6)–(7) writes $\mu_{rbc}(\phi, \operatorname{tr}\mathbf{D}_{rbc})$
   while the right-hand side of Eq. (7) uses the shear rate $(2\operatorname{tr}\mathbf{D}_{rbc}^2)^{1/2}$.
3. **Units of Eqs. (9a)–(9c):** Not explicitly printed alongside the equations (`not found` in equation text),
   but consistent with $\mu_f = 0.96\text{ cP}$ in §3, the polynomial values are in **cP (= mPa·s)** and
   $k = 11\text{ s}$: e.g. at $\phi = 0.45$, $\mu_0(0.45) = 60.02\text{ cP}$ ($0.060\text{ Pa}\cdot\text{s}$) and
   $\mu_\infty(0.45) = 4.74\text{ cP}$ ($4.74\text{ mPa}\cdot\text{s}$); at $\phi = 0.20$,
   $\mu_0(0.20) = 6.47\text{ cP}$ and $\mu_\infty(0.20) = 2.26\text{ cP}$; at $\phi = 0.30$,
   $\mu_0(0.30) = 19.41\text{ cP}$ and $\mu_\infty(0.30) = 3.18\text{ cP}$.
   (For $\phi \to 0$, note $\mu_0(\phi)$ goes slightly negative below $\phi \approx 0.00234$; for §3.1 at
   $\phi = 0.01$ hardened RBCs, Wu 2015 notes "the effect of deformability on viscosity was ignored".)
4. **Bulk viscosity constant $\beta_{20}$:** Listed among the 10 model parameters in §3, drops out in Eq. (13)
   under $\operatorname{tr}\mathbf{D}_{rbc} = 0$; numeric value `not found`.

---

### 3. Interphase Interaction Force — Drag, Lift & Diffusion (Wu 2015 §2.2)

Wu 2015 states: *"For the interaction force between the two components, we only consider terms which would
correspond to the Stokes drag force and Saffman's shear-lift force."*

$$\mathbf{f}_I
= \underbrace{\frac{9\mu_f}{2a^2}\,f(\phi)\,(\mathbf{v}_{rbc} - \mathbf{v}_f)}_{\text{Stokes drag with hindrance}}
+ \underbrace{\frac{3(6.46)(\rho_f \mu_f)^{1/2}}{4\pi a}\,\phi\,(2\operatorname{tr}\mathbf{D}_f^2)^{-1/4}\,\mathbf{D}_f(\mathbf{v}_{rbc} - \mathbf{v}_f)}_{\text{Saffman shear-lift}} \tag{10}$$

#### Drag law & hindrance function
- **Hindrance function** (Rusche & Issa, inline after Eq. 10):
  $$f(\phi) = \exp(2.68\,\phi) + \phi^{0.43}$$
- **Particle size $a$:** Wu 2015 text states *"where $a$ is the diameter of RBCs, here assumed as $8\,\mu\text{m}$."*
- **Transcription notes on Eq. (10):**
  - Standard Stokes drag per unit volume on spheres of radius $R = d/2$ at volume fraction $\phi$ is
    $\phi \frac{6\pi\mu_f R}{\frac{4}{3}\pi R^3} = \frac{9\mu_f \phi}{2R^2} = \frac{18\mu_f \phi}{d^2}$.
    Wu 2015 Eq. (10) prints $\frac{9\mu_f}{2a^2} f(\phi)$ (no explicit $\phi$ factor multiplying $f(\phi)$,
    and calls $a = 8\,\mu\text{m}$ the diameter rather than radius). Transcribed verbatim from PMC4521229.

#### Lift, diffusion, virtual mass, granular pressure, and packing limit
- **Lift:** Saffman shear-lift term (second term in Eq. 10, proportional to $\phi (\rho_f \mu_f)^{1/2} a^{-1} \dot{\gamma}_f^{-1/2} \mathbf{D}_f (\mathbf{v}_{rbc} - \mathbf{v}_f)$).
- **Diffusion / dispersion terms (Phillips migration, Brownian, turbulent dispersion):** **None** in Wu 2015
  (explicitly excluded — only Stokes drag + Saffman lift).
- **Virtual mass / Basset / Faxén forces:** **None** in Wu 2015.
- **Granular / collisional solid pressure:** **None** in Wu 2015.
- **Maximum packing limit $\phi_m$:** Discussed qualitatively in §2.1 (*"At a certain $\phi_m$ called the
  maximum packing fraction ... we have to set certain limits or restrictions on $\phi$ so that $\phi$ can
  never be greater than $\phi_m$"*), but **numerical value of $\phi_m$ is `not found`** in PMC4521229.

---

### 4. Phase Properties, Boundary Conditions & Benchmarks (Wu 2015 §3)

#### Material properties (§3)
| Symbol | Meaning | Value in Wu 2015 (`verified-primary`) |
|---|---|---|
| $\mu_f$ | Pure plasma dynamic viscosity | $0.96\text{ cP} = 0.96 \times 10^{-3}\text{ Pa}\cdot\text{s}$ |
| $\rho_f$ ($\rho_{f0}$) | Pure plasma density | $1027\text{ kg/m}^3$ |
| $\rho_{rbc}$ ($\rho_{rbc0}$) | Pure RBC density | $1093\text{ kg/m}^3$ ($\rho_{rbc}/\rho_f = 1.0643$) |
| $a$ | RBC diameter | $8\,\mu\text{m}$ |
| $\mu_0(\phi), \mu_\infty(\phi), k$ | RBC viscosity parameters | Eqs. (9a)–(9c), $k = 11$ |
| $f(\phi)$ | Drag hindrance function | $\exp(2.68\phi) + \phi^{0.43}$ |
| $\lambda_f$, $\beta_{20}$, $\phi_m$, $\mathbf{b}_f, \mathbf{b}_{rbc}$ | Bulk viscosities, packing limit, body forces | `not found` ($\lambda_f, \beta_{20}$ drop out for incompressible flow) |

#### Boundary conditions (Wu 2015 Table 1)
| Boundary | Pressure | Velocity | Volume Fraction |
|---|---|---|---|
| Inlet | Fixed flux (`fixedFluxPressure`) | Fixed value | Fixed value |
| Outlet | Fixed value (reference) | Zero gradient | Zero gradient |
| Wall | Fixed flux (`fixedFluxPressure`) | Fixed value ($0$) | Zero gradient |
| Symmetric Plane | `symmetryPlane` | `symmetryPlane` | `symmetryPlane` |

#### Benchmark cases simulated in Wu 2015 (§3.1–§3.4)
| Section | Experiment | Geometry | Inlet $\phi$ | Inlet $U$ (Re) | Key result |
|---|---|---|---|---|---|
| §3.1 | Karino & Goldsmith (1977) | Tubular sudden expansion $R_1 = 0.0755\text{ mm} \to R_2 = 0.252\text{ mm}$ ($d_1 = 151\,\mu\text{m}$, $E = 3.3377$), $L_{\text{in}} = 1\text{ mm}$, $L_{\text{out}} = 2\text{ mm}$ | $0.01$ (hardened RBCs) | $0.0757\text{ m/s}$ ($\text{Re}=12.2$), $0.233\text{ m/s}$ ($\text{Re}=37.8$) | $x_r = 230\,\mu\text{m}$ ($x_r/d_1=1.523$) and $715\,\mu\text{m}$ ($x_r/d_1=4.735$) |
| §3.2 | Patrick et al. | Rectangular microchannel, depth $100\,\mu\text{m}$ | $0.45$ | $350\,\mu\text{m/s}$ | RBC velocity profiles at $z = 8, 16\,\mu\text{m}$ |
| §3.3 | Zhao et al. (2008) | Rectangular sudden expansion, depth $100\,\mu\text{m}$ | $0.20$ | $0.167\text{ m/s}$, $0.833\text{ m/s}$ | Steady state at $t = 0.15\text{ s}, 0.035\text{ s}$; qualitative CFL & depletion zone (125,544 nodes) |
| §3.4 | Driven crevices | Rectangular channel with slots/crevices, depth $70\,\mu\text{m}$ | $0.40$ | $0.0238\text{ m/s}$ | Steady state at $t = 7\text{ s}$; RBC deficit in crevices |

---

### 5. Comparison: Wu et al. (2015) vs. 2023 COMSOL Model ([§6-comsol-2023](6-comsol-2023.md)) vs. Stock OpenFOAM v2306 `multiphaseEulerFoam`

| Item | Wu et al. (2015) (`verified-primary`, PMC4521229) | 2023 COMSOL Model ([§6-comsol-2023](6-comsol-2023.md), slides 6–7) | Stock OpenFOAM ESI v2306 `multiphaseEulerFoam` (M2b template) |
|---|---|---|---|
| **Framework / solver** | Custom two-fluid OpenFOAM solver (Mixture Theory, incompressible) | COMSOL "Euler-Euler Model, Laminar Flow" (two-fluid) | Stock `multiphaseEulerFoam` (incompressible multi-fluid PIMPLE, `constant/transportProperties`, `laminar`) |
| **Pressure coupling** | $-\nabla((1-\phi)p)$, $-\nabla(\phi p)$ (Eqs. 12–13) | Shared pressure $p$ (COMSOL Euler–Euler default) | Shared pressure $p_{rgh}$ / $p$ (`-(alpha_k/rho_k) grad(p_rgh)` in `pEqn.H`) |
| **Continuous phase (plasma)** | $\rho_f = 1027\text{ kg/m}^3$, $\mu_f = 0.96\text{ mPa}\cdot\text{s}$ (Newtonian) | $\rho = 1038\text{ kg/m}^3$, $\mu = 4.0\text{ mPa}\cdot\text{s}$ (unlabelled row on slide 6, most likely plasma/carrier — `TODO: confirm`) | Star units: $\rho_p^* = 1$, $\nu_p^* = 1/\text{Re}$ (Newtonian; exact scaling of plasma) |
| **Dispersed phase (RBC) density** | $\rho_{rbc} = 1093\text{ kg/m}^3$ ($\rho_{rbc}/\rho_f = 1.0643$) | $\rho = 1098\text{ kg/m}^3$ ($\rho_{rbc}/\rho_p = 1098/1038 = 1.0578$) | Star units: $\rho_{rbc}^* = \rho_{rbc}/\rho_p$ (configurable: $1.0643$ for Wu, $1.0578$ for 2023 COMSOL) |
| **Particle diameter** | $a = 8\,\mu\text{m}$ | $d_d = 8\,\mu\text{m}$ | `diameterModel constant; d = d_p / h1` (star units, e.g. $8/50 = 0.16$, $8/100 = 0.08$, $8/151 = 0.05298$) |
| **RBC viscosity / stress** | Yeleswarapu shear-thinning $\mu_{rbc}(\phi, \dot{\gamma}_{rbc})$ (Eqs. 7, 9a–9c) | Not shown on slides (based on Wu 2015 per owner) | **No stock $\phi$/$\dot{\gamma}$-dependent viscosity in ESI v2306 `multiphaseEulerFoam`** (`phaseModel` reads constant scalar `nu`). **Documented fallback:** constant effective RBC kinematic viscosity $\nu_{rbc}^* = (\mu_{rbc,\text{eff}}/\mu_p)/(\rho_r \text{Re})$ evaluated at inlet $\phi$ and nominal channel shear rate via Wu Eqs. (7, 9a–9c) (or explicit viscosity ratio `mu_ratio`) |
| **Drag law** | Stokes drag × Rusche & Issa hindrance $f(\phi) = \exp(2.68\phi) + \phi^{0.43}$ (Eq. 10) | Not shown on slides | **No stock Rusche–Issa hindrance.** Closest stock v2306 drag model: **`WenYu`** (Schiller–Naumann Stokes limit at low $Re_p$ × hindrance $(1-\phi)^{-2.65}$ via `(RBC plasma)` pair); optionally `SchillerNaumann` for dilute $\phi \le 0.01$ or `Gibilaro` / `GidaspowErgunWenYu` |
| **Lift force** | Saffman shear-lift (Eq. 10, 2nd term) | Not shown on slides | **None in stock ESI v2306 `multiphaseEulerFoam`** (`interfacialModels` only has `dragModels` & `heatTransferModels`; `fluid.lift(phase)` is commented out in `UEqns.H`). Documented limitation — flagged in `REVIEW_QUEUE.md` (requires `reactingTwoPhaseEulerFoam` or custom solver for lift) |
| **Diffusion / dispersion** | None | Not shown on slides | None (matches Wu 2015) |
| **Packing limit $\phi_m$** | Mentioned qualitatively; numeric value `not found` | Not shown on slides | MULES keeps $0 \le \alpha \le 1$, $\sum \alpha_k = 1$; no explicit $\phi_m < 1$ packing pressure in stock `multiphaseEulerFoam` |
| **Inlet BC** | $p$: fixedFluxPressure; $\mathbf{v}$: fixedValue; $\phi$: fixedValue | $u_c = u_{c0}, u_d = u_{d0} = 0.25\text{ m/s}$, $\phi_d = \phi_{d0}$ | `p_rgh`: `fixedFluxPressure`; `U.plasma`, `U.RBC`: `fixedValue` (parabolic or uniform); `alpha.RBC` = $\phi$, `alpha.plasma` = $1 - \phi$ (`fixedValue`, sum $= 1$) |
| **Wall BC** | $p$: fixedFluxPressure; $\mathbf{v} = 0$; $\phi$: zeroGradient | No slip $u_c = u_d = 0$ | `p_rgh`: `fixedFluxPressure`; `U.plasma`, `U.RBC`: `fixedValue (0 0 0)` (`noSlip`); `alpha.*`: `zeroGradient` |
| **Outlet BC** | $p$: fixedValue (ref); $\mathbf{v}$: zeroGradient; $\phi$: zeroGradient | "Pressure, no viscous stress", $p_0 = 0$ | `p_rgh`: `fixedValue 0`; `U.*`: `pressureInletOutletVelocity` (zeroGradient on outflow, prevents backflow instability); `alpha.*`: `zeroGradient` (or `inletOutlet` with `inletValue` = $\phi_{\text{in}}$) |
| **Validation target** | KG tube $d_1=151\,\mu\text{m}$, $E=3.3377$, $\phi=0.01$, $\text{Re}=12.2/37.8$, $x_r=230/715\,\mu\text{m}$; Zhao planar $\phi=0.20$ | Slide 7: 50 µm tube Re 3.37; 75 µm radius ($d_1=150\,\mu\text{m}$), Hct 0.22, Re 12.2, $x_r=224$ vs $230\,\mu\text{m}$ | Anchor suite in `campaign/m2b_design.jsonl`: KG Re 12.2/37.8 ($\phi=0.01$), Zhao G2 planar ($\phi \in \{0.2,0.3\} \times \text{Re} \in \{8.3,41.7\}$), G1 straight, G2 $E \in \{1.5, 3\}$ |

---

### 6. Lift in ESI v2306

**Source verification (`verified-primary` from ESI OpenFOAM-v2306 C++ source):** Inspected verbatim via the official
OpenFOAM v2306 API source (`api.openfoam.com/2306/`) and OpenFOAM-v2306 tree under
`src/phaseSystemModels/reactingEuler/multiphaseSystem/interfacialModels/liftModels/`,
`src/phaseSystemModels/reactingEuler/multiphaseSystem/phasePair/phasePair.C`,
`MomentumTransferPhaseSystem.C`, and
`applications/solvers/multiphase/reactingTwoPhaseEulerFoam/pU/pEqn.H`.

#### 6.1 Base lift formulation & sign convention in `reactingTwoPhaseEulerFoam`

In `liftModel.C` (lines 95–109), the base class computes:

$$\mathbf{F}_i = C_L\,\rho_c\,\big(\mathbf{U}_r \times (\nabla \times \mathbf{U}_c)\big), \qquad
\mathbf{F} = \alpha_d\,\mathbf{F}_i = \alpha_d\,C_L\,\rho_c\,\big((\mathbf{U}_d - \mathbf{U}_c) \times (\nabla \times \mathbf{U}_c)\big)$$

where `pair_.Ur() = dispersed().U() - continuous().U()` (`phasePair.C` line 124).

In `MomentumTransferPhaseSystem.C` (lines 512–529), `phiFs()` assigns `+fvc::flux(rAU1 * F)` to `phase1`
(dispersed, `RBC`) and `-fvc::flux(rAU2 * F)` to `phase2` (continuous, `plasma`). In `pU/pEqn.H` (lines 79,
89, 126, 132), `phiHbyA1 = fvc::flux(HbyA1) - phigF1 - ...` where `phigF1` adds `+phiF1`. Therefore the net
lift force acting **on the dispersed RBC phase** is:

$$\mathbf{F}_{\text{lift on RBC}}^{\text{OF}} = -\mathbf{F} = -\alpha_d\,C_L\,\rho_c\,\big((\mathbf{U}_d - \mathbf{U}_c) \times (\nabla \times \mathbf{U}_c)\big)$$

**Sign check (2D planar channel $y \in [0, 1]$, Poiseuille flow $u_c(y) > 0$, bottom wall at $y = 0$):**
Near the bottom wall, $\dot{\gamma} = \partial u_c / \partial y > 0$, so
$\nabla \times \mathbf{U}_c = (0, 0, -\dot{\gamma})$. For a lagging RBC ($U_{r,x} = U_{d,x} - U_{c,x} < 0$),
$\mathbf{U}_r \times (\nabla \times \mathbf{U}_c) = (0, +U_{r,x}\dot{\gamma}, 0)$, giving
$F_{\text{lift on RBC}, y}^{\text{OF}} = -\alpha_d C_L \rho_c U_{r,x} \dot{\gamma} > 0$ for **$C_L > 0$**
(away from the wall toward the channel centreline). This matches the exact sign of Wu 2015 Eq. (13):
for simple shear $D_{f,xy} = \frac{1}{2}\dot{\gamma}$,
$F_{\text{lift on RBC}, y}^{\text{Wu}} = -\frac{3(6.46)(\rho_f\mu_f)^{1/2}}{8\pi a}\phi\dot{\gamma}^{1/2} U_{r,x} > 0$
when $U_{r,x} < 0, \dot{\gamma} > 0$.

#### 6.2 Complete inventory of `Foam::liftModels` in ESI OpenFOAM-v2306

`reactingTwoPhaseEulerFoam` provides **exactly 6** runtime-selectable `liftModel` types (confirmed from
`namespaceFoam_1_1liftModels.html` and the `.H`/`.C` files; **`SaffmanMei` does NOT exist** in ESI v2306
Euler–Euler — `SaffmanMei` exists only in OpenFOAM Foundation's `multiphaseEuler` or in ESI's Lagrangian
`SaffmanMeiLiftForce`):

| `TypeName` (dict `type`) | Class / source file | Exact C++ formula for $C_L$ | References cited in header | Microfluidic RBC suitability |
|---|---|---|---|---|
| `none` | `noLift` (`noLift.C`) | $C_L = 0$, $\mathbf{F} = \mathbf{0}$ | — | Disables lift (equivalent to M2b-A) |
| `constantCoefficient` | `constantLiftCoefficient` (`constantLiftCoefficient.C`) | $C_L = \text{const}$ (dict key `Cl`) | — | **Usable (secondary option)**: can be matched to Saffman at a nominal shear rate $\dot{\gamma}^*_{\text{nom}}$ (gives $\propto \dot{\gamma}^1$ instead of $\dot{\gamma}^{1/2}$) |
| **`LegendreMagnaudet`** | `LegendreMagnaudet` (`LegendreMagnaudet.C`) | $\text{Re}_p = \max(\text{pair.Re}(), \text{residualRe})$, $\text{Sr} = \frac{d_p^2}{\text{Re}_p \nu_c}\|\nabla \mathbf{U}_c\|$, $C_{L,\text{low}}^2 = \frac{(6\cdot 2.255)^2 \text{Sr}^2}{\pi^4 \text{Re}_p (\text{Sr} + 0.2\,\text{Re}_p)^3}$, $C_{L,\text{high}}^2 = \left(0.5\frac{\text{Re}_p + 16}{\text{Re}_p + 29}\right)^2$, $C_L = \sqrt{C_{L,\text{low}}^2 + C_{L,\text{high}}^2}$ | Legendre & Magnaudet (1998), *J. Fluid Mech.* 368, 81–126; Otromke (2013) PhD thesis | **Selected default for M2b-B**: analytically reduces to exact Saffman $\dot{\gamma}^{1/2}$ shear-lift scaling in our low-$\text{Re}_p$ regime ($\text{Sr} \gg 0.2\,\text{Re}_p$) |
| `Moraga` | `Moraga` (`Moraga.C`) | $\text{Re}_p$ clamped to $[1200, 18800]$, $\text{Sr}^2 = \frac{d_p^2}{\nu_c}\|\nabla \mathbf{U}_c\|$ clamped to $[0.0016, 0.04]$; $C_L = 0.2\exp(-\text{Re}_p\text{Sr}^2/3.6\times 10^5 - 0.12)\exp(\text{Re}_p\text{Sr}^2/3.0\times 10^7)$ | Moraga, Bonetto & Lahey (1999), *Int. J. Multiphase Flow* 25(6–7), 1321–1372; Otromke (2013) | **Unusable**: turbulent high-$\text{Re}_p$ correlation; clamps all microfluidic cells to $\text{Re}_p = 1200$ |
| `Tomiyama` | `TomiyamaLift` (`TomiyamaLift.C`) | $Eo_H = |\rho_d - \rho_c|\,|\mathbf{g}|\,d_H^2/\sigma$; $f(Eo_H) = 0.0010422\,Eo_H^3 - 0.0159\,Eo_H^2 - 0.0204\,Eo_H + 0.474$; $C_L = \min(0.288\tanh(0.121\,\text{Re}_p), f)$ for $Eo_H<4$, $f$ for $4\le Eo_H < 10.7$, $-0.288$ for $Eo_H \ge 10.7$ | Tomiyama et al. (2002), *Chem. Eng. Sci.* 57(11), 1849–1858 | **Unusable**: buoyancy/surface-tension bubble correlation; at $\mathbf{g}=0$, $Eo_H=0$ and $C_L \approx 0.035\,\text{Re}_p \to 0$ |
| `wallDamped` | `wallDamped` (`wallDampedLift.C`) | Wrapper multiplying an inner `lift` submodel by a `wallDamping` model | — | Optional decorator (not needed for Saffman wall repulsion) |

#### 6.3 Analytic reduction of `LegendreMagnaudet` to Saffman shear lift & star-unit coefficients

In microfluidic RBC flows, the slip Reynolds number is tiny ($\text{Re}_p \ll 1$) while the shear Reynolds number
$\text{Re}_G = d_p^2 \|\nabla \mathbf{U}_c\| / \nu_c = \text{Re}\,(d_p^*)^2\,\dot{\gamma}^*$ is $\mathcal{O}(0.1\text{--}10)$,
so $\text{Sr} = \text{Re}_G / \text{Re}_p \gg 0.2\,\text{Re}_p$. In this limit (`LegendreMagnaudet.C` lines 82–91):

$$C_{L,\text{low}} \longrightarrow \frac{6 \times 2.255}{\pi^2}\,\frac{1}{\sqrt{\text{Re}_p\,\text{Sr}}}
= \frac{6 \times 2.255}{\pi^2}\,\frac{1}{\sqrt{\text{Re}_G}}
= \frac{6 \times 2.255}{\pi^2}\,\frac{\nu_c^{1/2}}{d_p\,\|\nabla \mathbf{U}_c\|^{1/2}}$$

Notice that $\text{Re}_p$ cancels identically from $\text{Re}_p\,\text{Sr} = \text{Re}_G$! Substituting $C_{L,\text{low}}$
into `liftModel::F()` for simple shear ($\|\nabla \mathbf{U}_c\| = |\nabla \times \mathbf{U}_c| = \dot{\gamma}$)
yields:

$$\big|\mathbf{F}_{\text{LM, low-Re}}\big|
= \phi\,\underbrace{\frac{6 \times 2.255}{\pi^2}}_{1.3709}\,\frac{(\rho_c \mu_c)^{1/2}}{d_p}\,\dot{\gamma}^{1/2}\,|U_r|$$

Compare this exact Saffman scaling across the three prefactors:
- **Classic Saffman (1965) rigid sphere** ($J(\infty) = 2.255$, $6.46 = \frac{9}{\pi}J(\infty)$, sphere radius $R = d_p/2$):
  $K_{\text{Saff}} = \frac{3(6.46)}{2\pi} = \frac{13.5 \times 2.255}{\pi^2} = 3.0844$.
- **ESI v2306 `LegendreMagnaudet` low-$\text{Re}_p$ limit** (clean spherical bubble, exact $(2/3)^2 = 4/9$ of rigid-sphere Saffman):
  $K_{\text{LM}} = \frac{6 \times 2.255}{\pi^2} = 1.3709$.
- **Wu et al. (2015) Eq. (10) verbatim printed form** (uses diameter $a = d_p$ and $\mathbf{D}_f \mathbf{U}_r = \frac{1}{2}\dot{\gamma} U_r$):
  $K_{\text{Wu}} = \frac{3(6.46)}{8\pi} = 0.7711$.

Thus **`LegendreMagnaudet`** (with `residualRe 1e-3`, so $0.2\,\text{Re}_{\text{res}}^2 = 2\times 10^{-7} \ll \text{Re}_G$ everywhere shear is active while keeping $C_L$ bounded near the zero-shear centreline) recovers the **exact $\phi (\rho_c \mu_c)^{1/2} d_p^{-1} \dot{\gamma}^{1/2} U_r$ Saffman shear-lift scaling locally at every cell** with a prefactor ($1.3709$) right between Wu 2015's printed Eq. (10) ($0.7711$) and rigid-sphere Saffman ($3.0844$) — and requires no ad-hoc nominal shear rate.

**Star-unit mapping (`cfd/gen/m2b_lift.py`, M2b-B):**
- Scales identical to M2b-A: $h_1^* = 1$, $U_{\text{mean}}^* = 1$, $\rho_p^* = 1$, $\mu_p^* = 1/\text{Re}$, $\rho_{rbc}^* = \rho_r$, $\mu_{rbc}^* = \mu_r/\text{Re}$, $d_p^* = d_p/h_1$.
- Isothermal constant-density `heRhoThermo` (`equationOfState rhoConst`, `transport const`, `thermo eConst`, `nEnergyCorrectors 0`), `blending { default { type none; continuousPhase plasma; } }`, `WenYu` drag on `(RBC in plasma)`, `LegendreMagnaudet` lift on `(RBC in plasma)` (default).
- Optional `constantCoefficient` Saffman-matched derivation in star units: equating
  $\phi C_L \rho_p^* |U_r^*| \dot{\gamma}^* = \phi K (\text{Re})^{-1/2} (d_p^*)^{-1} (\dot{\gamma}^*)^{1/2} |U_r^*|$
  at nominal shear rate $\dot{\gamma}^*_{\text{nom}}$ (e.g. mean Poiseuille shear $\dot{\gamma}^*_{\text{nom}} = 3$ or wall shear $6$) yields
  $$C_{L,\text{Saff}}^* = \frac{K}{d_p^*\,\sqrt{\text{Re}\,\dot{\gamma}^*_{\text{nom}}}} = \frac{K}{\sqrt{\text{Re}_{G,\text{nom}}}}$$
  with $K = K_{\text{Wu}} = 0.7711$ (Wu Eq. 10) or $K = K_{\text{Saff}} = 3.0844$ (rigid-sphere Saffman). Both `LegendreMagnaudet` (default) and `constantCoefficient` are implemented in `cfd/gen/m2b_lift.py`.

---

## Steady 2D two-fluid (Euler–Euler) formulation for the FEniCSx solver (V3 / V4)


Status: DRAFT v1 — formulation for review before code (AGENTS rule 7).
Sources: COMSOL 6.3 CFD Module User's Guide, "The Euler–Euler Model Equations" and "Some Notes on Boundary
Conditions" (doc.comsol.com/6.3/doc/com.comsol.help.cfd/cfd_ug_fluidflow_multi.09.158.html and .09.159.html), equation
images read directly 2026-11-24 → `verified: COMSOL 6.3 docs, image no.` below; Wu et al. 2015 closures →
[§2-theory](2-theory.md) (verified-primary).

### 1. COMSOL Euler–Euler model as documented (verified, 09.158 images 01–55)
Phases: continuous c (plasma), dispersed d (RBC); φ_c = 1 − φ_d (img 03).
- Continuity, incompressible, no mass transfer: ∇·(φ_d u_d) = 0 (img 05) solved for φ_d;
  mixture ∇·(φ_d u_d + (1 − φ_d) u_c) = 0 (img 06) solved for the shared pressure p.
- Momentum, non-conservative, volume fraction **outside** the divergence (img 07, 08):
  ρ_k φ_k [∂u_k/∂t + u_k·∇u_k] = −φ_k ∇p + φ_k ∇·τ_k + φ_k ρ_k g + F_m,k + φ_k F_k.
  Implemented **divided by φ_k** (img 14, 15): ρ_k(u_k·∇u_k) = −∇p + ∇·τ_k + F_m,k/φ_k + F_k,
  with 1/φ_k → 1/φ_k,pos, φ_k,pos = max(φ_k, φ_k,res), φ_res = 1e-4 default (img 54); drag and viscosity use
  φ_k,reg = min(max(φ_k, 0), 1) (img 55). Consequence: u_d stays defined where φ_d → 0 (09.159 text).
- Optional solid pressure for ρ_d ≫ ρ_c: −∇p_s, ∇p_s = −G(φ_c)∇φ_c, G = 10^(B₁φ_c + B₂) (img 09, 46–50).
- Viscous stress τ_k = μ_k^m(∇u_k + ∇u_kᵀ − ⅔(∇·u_k)I) (img 10, 11). Default μ_c^m = μ_d^m = μ_mix (img 17);
  options: Krieger μ_mix = μ_c(1 − φ_d/φ_max)^(−2.5 φ_max) (img 18; φ_max 0.62 solids), droplet variant (img 19),
  linear μ_mix = φ_c μ_c + φ_d μ_d (img 20).
- Drag F_drag,c = −F_drag,d = β u_slip, u_slip = u_d − u_c (img 23, 24); β = (3/(4d)) φ_d C_D ρ_c |u_slip| (img 26),
  φ_c-weighted variant (img 27); Gidaspow = Wen–Yu (img 28) / Ergun (img 29); Schiller–Naumann (img 31);
  Hadamard–Rybczynski (img 32); Stokes C_d = 24/Re_p (img 33) ⇒ β = 18 μ_c φ_d / d²; Ishii–Zuber with mixture
  viscosity Re_m (img 34–37).
- **No predefined lift, wall lubrication or dispersion** — extra forces enter as volume forces F_k.
- Boundary notes (09.159): where φ_d = 0 at an inlet set u_d = u_c; prescribe φ_d only where nothing is transported
  toward the inlet.

### 2. Our formulation (star units: h₁, U = mixture mean velocity, μ_ref = μ_p plasma, ρ_ref = ρ_p)
Unknowns u (plasma), w (RBC), p, φ (= φ_d). Steady. Re_p = ρ_p U h₁/μ_p, ρ̂ = ρ_rbc/ρ_p = 1093/1027 (Wu).
- φ:      ∇·(φ w) = 0 (+ V4 dispersion flux, §3)
- mixture: ∇·(φ w + (1 − φ) u) = 0
- plasma:  Re_p (u·∇)u = −∇p + ∇·(2 μ̂_c D(u)) + [K(w − u) + L]/(1 − φ)_pos
- RBC:     Re_p ρ̂ (w·∇)w = −∇p + ∇·(2 μ̂_d D(w)) − [K(w − u) + L]/φ_pos (+ V4 terms)
(COMSOL's divided form; the ⅔∇·u term is dropped as each phase is incompressible.)

Closures (star units, d̂ = d_RBC/h₁, d = 8 µm per Wu):
| Term | V3 `EE-DL` | V4 `EE-full` |
|---|---|---|
| drag K | Stokes with Wu hindrance: K = 18 φ f(φ)/d̂², f = exp(2.68φ) + φ^0.43 ([§2-theory](2-theory.md) §3; standard per-volume Stokes form, see 14 transcription note) | same |
| lift L | Wu/Saffman: L = C_S φ √Re_p /d̂ · γ̇_u^(−1/2) D(u)(w − u), C_S = 3·6.46/(4π) (Wu Eq. 10, nondimensionalised) | same |
| viscosity | μ̂_c = μ̂_d = μ_mix/μ_p, Newtonian high-shear mixture: μ_mix = μ_p(1 − φ) + μ_∞(φ) (Wu Eqs. 8, 9b) | shear-thinning RBC phase: Wu Eq. 7 (μ₀(φ), μ_∞(φ), k = 11 s) in the RBC equation, plasma μ_p |
| wall lift | — | deformable-cell wall lift (`TODO: source`, candidate form U_lift ∝ γ̇ a³/y²) |
| shear-induced dispersion | — | Phillips flux ([§2-theory](2-theory.md) §5.2 constants, `secondary`) |
| virtual mass | — | C_vm = 0.5, C_vm φ ρ_p [(w·∇)w − (u·∇)u] (`recalled`, classical sphere value) |
Regularisations: γ̇_u → √(2D:D + ε²), ε = 1e-3; φ_pos = max(φ, 1e-4).

### 3. Discretisation (decisions)
Taylor–Hood P2 (u, w) / P1 (p), P1 φ with SUPG (COMSOL also uses streamline diffusion); monolithic Newton
(PETSc SNES + MUMPS), continuation in φ_in then Re_p. Quadrilateral meshes built from the same `cfd/gen/geometry.py`
blocks as the OpenFOAM cases (same-mesh pairing with V1/V2).
BCs: inlet u, w, φ = developed profiles from a 1D fully-developed solve (owner choice: developed inlet);
walls u = w = 0 (Wu Table 1: fixed value 0 both phases), no φ flux; outlet p = 0, natural (do-nothing) for u, w, φ.

### 4. Open points (review)
1. Wu Eq. 10 drag lacks an explicit φ (and calls a = 8 µm a diameter) — we use the standard per-volume Stokes form;
   sensitivity run with Wu's printed form.
2. Wu's stress-factor-½ ([§2-theory](2-theory.md) note 1) applies to V4's Eq. 7 viscosity: keep printed form (as T6 decision).
3. Wall-lift closure for V4 needs a source before V4 production.

### 5. COMSOL solver & stabilisation as documented (read 2026-11-24, doc.comsol.com 6.3) — exactness map
Sources: CFD Module "Pseudo Time Stepping for Laminar Flow Models" (cfd_ug_fluidflow_single.06.100, Eqs. 3-60, 3-61);
Reference Manual "About the Stationary Solver" (comsol_ref_solver.36.118, damped Newton); "Stabilization Techniques"
(comsol_ref_modeling.19.090) + references (19.091). Equations read from the page images → `verified`.
| COMSOL feature | Documented form | Our status |
|---|---|---|
| Pseudo time stepping | ρ(u − nojac(u))/Δt̃ added to the stationary equations; Δt̃ = CFL_loc·h/|u| per element; one pseudo step per nonlinear iteration | **can be exact** |
| CFL schedule (Eq. 3-61) | CFL_loc = 1.3^min(k,9) + [k>20]·9·1.3^min(k−20,9) + [k>40]·90·1.3^min(k−40,9), k = nonlinear iteration (default manual expression; automatic = PID "CFLCMP", details not read) | **can be exact** (manual expression) |
| Damped Newton | affine-invariant damped Newton (error estimate E from f'(U₀)E = −f(U₁); reduce λ until error decreases) | can be reimplemented (~30 lines); PETSc line search is only equivalent |
| Streamline diffusion | GLS without viscous terms in the test operator (reduces to SUPG); τ not given on this page | near-exact once the fluid-flow stabilisation page (comsol_ref_fluidflow.29.18) is read |
| Crosswind diffusion | weak term −ν^h ∂c/∂x_i g^{ij} ∂c/∂x_j (Hauke & Hughes 1998 form), ν^h = **modified** Hughes–Mallet / do Carmo–Galeão 1991, × (√2)^(1−N) | **not exact**: COMSOL's modification is not documented → closest published formula |
| Discretisation | Euler–Euler default P1+P1 with streamline diffusion (09.081); P2+P1 option | we use P2+P1 (Taylor–Hood) |

---

## Cell-free layer (CFL) modelling — literature read and V4 closure proposal


Status: DRAFT. Owner rule: no corpus launch until V4 is settled.

### Why V3/V4-as-built fail (evidence)
Steady FEniCSx V4 (drag + Saffman lift, COMSOL pseudo-time recipe) converges with only 1 % wall depletion; OpenFOAM
EE (WenYu + LegendreMagnaudet) shows no Sezer ΔP* minimum; the owner's 2023 COMSOL time-dependent runs showed a CFL
only in early snapshots ([§6-comsol-2023](6-comsol-2023.md)). Saffman lift ∝ slip, and RBCs barely slip → no sustained CFL.

### Literature (read 2026-11-24 via NCBI PMC full text unless noted)
| Source | Access | What it says (verified unless marked) |
|---|---|---|
| Fedosov, Caswell, Popel, Karniadakis, *Microcirculation* 17:615 (2010), doi 10.1111/j.1549-8719.2010.00056.x, PMC3529161 | full | DPD deformable RBCs, tubes D = 10–40 µm, Ht 0.15–0.45: RBCs migrate to the centre; **CFL wider for lower Ht and larger D**; D = 10 µm, Ht 0.45: CFL 1.19 / 1.30 / 1.45 µm (three wall models); CFL **slightly decreases with shear rate** (consistent with Kim et al. 2007 for 100–500 s⁻¹); in vivo: Maeda 1–1.8 µm (D 10–40 µm, Ht 0.45), Kim 0.5–3 µm (D 10–50 µm, Ht 0.42), Yamaguchi ≈ 4 µm; Pries Fåhræus relation Ht/Hd = Hd + (1 − Hd)(1 + 1.7e^(−0.35D) − 0.6e^(−0.01D)) (Eq. 12, D in µm); full CFL(D, Ht) curves in Fig. 10 (figure only — digitise) |
| Abkarian & Viallat, *Biophys J* 89:1055 (2005), PMC1366591 | abstract only | tank-treading non-spherical vesicles near a wall feel a **viscous lift force** (up to 150 pN), functional dependence on wall distance, wall shear rate, viscosity, size, reduced volume; deformable circulating cells feel it too. Exact formula: in the paper body (`TODO`: owner PDF) |
| Abkarian, Lartigue, Viallat, *PRL* 88:068103 (2002), doi 10.1103/physrevlett.88.068103 | metadata | "determination of the lift force" (recalled: lift ∝ μ γ̇ R³/y² for vesicles) |
| Kumar & Graham, *PRL* 109:108102 (2012), doi 10.1103/physrevlett.109.108102 | metadata | margination mechanism = wall-induced migration of deformable particles + shear-induced pair collisions (recalled content) |
| Hariprasad & Secomb, *Phys Rev E* 92:033008 (2015), PMC4765958 | full | 2D deformable RBC in Stokes Poiseuille flow (10–14 µm channels): **non-inertial lateral migration and focusing**, caused by wall proximity and velocity-profile curvature; rigid particles cannot migrate in Stokes flow (reversibility) |
| Kim, Kong, Popel, Intaglietta, Johnson, *Am J Physiol Heart* (2007), doi 10.1152/ajpheart.01090.2006 | metadata | CFL width in arterioles, temporal/spatial variation |

Take-aways: (1) the CFL is sustained by **deformability-induced lateral migration** (wall + curvature), balanced by
**cell–cell collisions**; rigid-sphere Saffman lift is not the mechanism. (2) In vessels ≤ 50 µm the CFL is
O(1–5 µm); it decreases with Ht. (3) Data for our device regime (h 80–150 µm, Ht 0.01–0.1) are sparse: Gracka 2022
(Ht 5 %: 5 and 35 µm in hyperbolic contractions), Zhao 2008 via Sezer (Ht ≤ 0.20: depleted near-wall regions larger
than the cell size).

### Proposed V4 closure (for review)
Add to the RBC phase / φ equation:
1. **Wall-induced migration** (deformability): lateral velocity v_L = C_L γ̇ a³ / y_w² · n_w (a = RBC radius, y_w wall
   distance, n_w unit normal away from the nearest wall), entered as the force that drives this velocity against
   drag, F_L = β v_L (slip-independent). Form: Abkarian-type (`TODO`: verify exponent in Abkarian 2005 full text).
2. **Shear-induced dispersion** (collisions): Phillips flux N = −K_c a² φ ∇(γ̇ φ) ([§2-theory](2-theory.md) §5.2; K_c secondary).
3. **1D steady balance near a flat wall** (γ̇ ≈ const): φ v_L = K_c a² γ̇ φ dφ/dy ⇒ dφ/dy = (C_L a / K_c) / y² ⇒
   φ(y) = φ_∞ − (C_L a/K_c)/y for y > δ, φ = 0 inside, with **δ = (C_L / K_c) · a / φ_∞**.
   Predictions: shear rate cancels (CFL ≈ independent of flow rate: matches Sezer §3.2, Bernardelli's flat loading
   30→50 µL/min, Fedosov's weak shear dependence); δ ∝ 1/φ (wider CFL at low Ht: matches Fedosov, Zhao, Gracka).
   One calibration constant C_L/K_c. Example: if δ = 2.5 µm at Ht 0.45 (to be read from Fedosov Fig. 10, D = 40 µm),
   a = 4 µm ⇒ C_L/K_c ≈ 0.28 ⇒ δ ≈ 22 µm at Ht 0.05 (device regime) — to be checked against Gracka (5 %).
4. Validation targets: Fedosov Fig. 10 (digitised), Kim 2007, Maeda (via Fedosov), Gracka 2022 (needs PDF), Zhao/Sezer
   qualitative depletion. Gate: the CFL must **persist** at steady state.
Numerics: wall distance from a Poisson/Eikonal solve per mesh; near-wall mesh grading so δ (≥ 1–2 µm) spans ≥ 4 cells.

### Update 2026-11-24 — owner PDFs read in full (knowledge/papers/pdf not committed)
**Abkarian & Viallat 2005, Biophys J 89:1055 — `verified`.** Giant vesicles (R 8.7–67 µm) near a wall, wall shear
0.1–10 s⁻¹. Measured hovering height h grows linearly with γ̇ (Fig. 7, Eq. 6) ⇒ lift **F_l = η γ̇ R³ f(ν) / h** (Eq. 4),
h = membrane–wall gap, ν = reduced volume, f(1−ν) empirical (Fig. 8: rises from ~0 at ν→1 to ~1.25–1.75 for
1−ν ≳ 0.05); valid above γ̇_c where the shape is stationary; measured h ≈ 1–15 µm (h₀ = 2.2 ± 0.4 µm at unbinding).
Discussion: for RBCs ν = 0.7, **f ≈ 2 (authors' extrapolation)**, R = 3 µm → 31–155 pN at wall stress 0.2–1 Pa and
h = 350 nm; "the discoid shape of RBCs ... can induce a strong lift force, which helps to maintain the cells away from
the endothelial walls producing a cell-free layer". Far-field (h ≫ R) drift velocity computed by Olla 2000 (Physica A
278:87, cited as ref. 15; content not read).
**Gracka et al. 2022, CMPB 226:107117 (CC BY) — `verified`.** Hyperbolic contractions (Rodrigues 2016 chips): throat
54 µm (A, e_H = 2) / 20 µm (B, e_H = 3), main channel 400 µm wide, depth 50 µm, ports 12 mm apart; fluid Dx40
ρ 1042 kg/m³, μ 5.2 mPa·s; RBC ρ 1185, d 8 µm, Ht 5 %; Q 10 µL/min, inlet 0.0025 m/s, Re < 0.1. EE (Fluent): Wen–Yu
drag (Eq. 6–7), Jung Carreau–Yasuda RBC viscosity (Eq. 11–13), KTGF bulk viscosity (Eq. 14–15), **no lift**; E-L DPM
one-way. **Experimental CFL ≈ 5 µm (A) and 35 µm (B), ~300 µm downstream of the throat** (CFL edge = φ > 2 %);
EE CFL mesh-dependent (A: 8/7/5/5 µm; B: 40/40/35/35 µm, Table 1); CFL φ < 2 % vs 6–8 % core. Caveat: depth (50 µm) <
width (400 µm), so a planar 2D model in the width plane is a Hele-Shaw-type approximation.

### V4 closure (revised, replaces the C·γ̇a³/y² guess above)
Wall lift per unit volume on the RBC phase (Abkarian Eq. 4 divided by cell volume, R cancels):
  **f_wl = C_A φ μ_c γ̇ / y_gap · n_w,  C_A = 3 f(ν)/(4π) ≈ 0.48 (f = 2, ν = 0.7)**,
y_gap = wall distance − R (floored at 0.1 R), n_w = unit vector away from the wall; far-field option (y_gap > R):
× R/y_gap (Olla-type decay, `recalled`) — run both, report the difference. Equal and opposite on plasma (interphase).
Counter-flux: Phillips shear-induced dispersion (K_c 0.41, K_μ 0.62, secondary). **No fitted constant.**
Validation: Gracka A/B (5/35 µm, Ht 5 %), Fedosov trends (CFL ↑ as Ht ↓; weak shear dependence), Zhao depletion;
gate = CFL persists at steady state.
