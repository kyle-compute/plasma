# Plasma Sputtering Simulator

## 1. Goal

Build a GPU-accelerated plasma simulation for **High Power Impulse Magnetron Sputtering (HiPIMS)** that operates at two fidelity levels:

1. **0D Global Model (Ionization Region Model)** — Volume-averaged ODE system tracking species densities and electron temperature in the ionization region. Runs in seconds. Current status: reduced Cu/Ar IRM prototype with numerical sanity-check tests (not yet reproducing specific published figures/tables from Gudmundsson 2022 or Brenning 2021).

2. **2D Axisymmetric PIC-MCC** — Fully kinetic Particle-in-Cell with Monte Carlo Collisions in cylindrical (r, z) geometry. Tracks individual macro-particles for electrons, Ar+, metal ions, and neutrals. Self-consistent E-field via Poisson solver, prescribed static B-field from magnetron geometry. Runs on GPU. Used for spatially-resolved plasma dynamics such as sheath structure, exploratory substrate IEDFs, and discharge-geometry trends. Because the model is axisymmetric in `(r, z)`, it cannot resolve azimuthal spoke formation.

The codebase is designed to be **extensible toward AI-driven plasma control** — specifically, a future pipeline where computer vision observes real plasma behavior (e.g., plasma globes, OES diagnostics) and ML surrogates trained on simulation data close the loop between prediction and experiment.

### Target outputs (current status)

- Ionization probability (`alpha_t`) and transport metrics (`xi_t`) are emitted by the 0D model. `beta_t` and related metal-transport closures are still partly heuristic and should not be treated as benchmark observables yet.
- Ion energy distribution function (IEDF) at the substrate is now collected from absorbed boundary-crossing positive ions in the PIC runtime. It is model-derived for the species actually simulated; it is not yet a fully benchmarked deposition diagnostic.
- Ionized flux fraction as a function of pulse parameters.
- Mixed HiPIMS + dcMS pulsing optimization.
- Energy cost per deposited ion (`epsilon_ti`) remains derived from closure quantities and is not validation-grade.

---

## 2. Hardware

| Component | Spec | Role |
|-----------|------|------|
| **GPU** | NVIDIA RTX 4080 Super | 16 GB GDDR6X, 10240 CUDA cores, ~52 TFLOPS FP32. Development, 0D model, small PIC runs (up to ~10M particles 2D) |
| **CPU** | Intel i9-10850K | 10C/20T, 3.6 GHz base / 5.2 GHz boost. ODE integration, data pipeline, preprocessing |
| **RAM** | 32 GB DDR4 | Sufficient for all preprocessing, cross-section tables, and post-processing |
| **Cluster** (on-demand) | 4x NVIDIA H100 80GB HBM3 | ~10 hours available. Production PIC runs: 50-200M particles, full pulse cycles, multi-GPU domain decomposition |

### Hardware constraints that shape design

- **16 GB VRAM is the primary bottleneck for PIC.** A single particle in 2D(r,z) PIC needs ~56 bytes (position r,z + velocity vr,vz,vtheta + weight + species tag). 16 GB fits ~280M particles raw, but field arrays, charge density grids, and collision tables eat ~2-4 GB, so practical limit is ~100-200M particles with careful memory management. For HiPIMS densities (~10^18-10^19 m^-3), a 2D domain of ~10 cm x 5 cm with macro-particle weight tuning can work within budget.
- **DDR4 bandwidth (25-50 GB/s) vs HBM3 (3.35 TB/s on H100)** means the H100 cluster is ~30-60x faster for memory-bound PIC operations. Use 4080 for development, H100 for production.
- **10 hours on 4x H100** is enough for ~5-10 full HiPIMS pulse simulations (100 us pulse + 200 us afterglow) at publication-quality resolution, or one large parameter scan on the 0D model (~10,000 configurations).

---

## 3. Physics Model

### 3.1 Species

From Gudmundsson 2022 IRM for Cu/Ar discharge (extensible to Ti/Ar, Al/Ar).

**Current implementation status:** The 0D model now carries dual electron populations and an expanded Cu/Ar state including argon excited states and copper excited/metastable channels. That said, several transport observables are still closure-driven and the full state below should be read as the intended physical basis, not a claim that every listed observable is already benchmark-calibrated.

| Species | Symbol | Notes |
|---------|--------|-------|
| Electrons (cold) | e_cold | Maxwellian, T_e = 1-7 eV |
| Electrons (hot) | e_hot | Secondary electrons, 200-1000 eV |
| Argon ground state (cold) | Ar_c | Feedstock gas at gas temperature |
| Argon ground state (hot) | Ar_h | Returned from target after ion impact, ~few eV |
| Argon ground state (warm) | Ar_w | Implanted in target, leave at ~0.1 eV |
| Argon metastable | Ar_m | Ar(4s[3/2]_2) at 11.55 eV |
| Argon resonant | Ar_r | Ar(4s'[1/2]_0) at 11.72 eV |
| Argon 4p levels | Ar_4p | Effective combined level |
| Ar+ | Ar+ | Single ionization |
| Ar2+ | Ar2+ | Double ionization |
| Metal ground state | M (Cu/Ti) | Sputtered from target, ground state |
| Metal metastable 1 | M_m1 | e.g. Cu(3d^9 4s^2 ^2D_{5/2}) at 1.39 eV |
| Metal metastable 2 | M_m2 | e.g. Cu(3d^9 4s^2 ^2D_{3/2}) at 1.64 eV |
| Metal excited | M_ex | Combined excited levels |
| M+ | M+ | Single ionization |
| M2+ | M2+ | Double ionization |

### 3.2 Reaction Set

27 reactions from Gudmundsson Table 1, grouped:

**Electron-Argon (R1-R9):**
- R1: e + Ar(3p^6) -> Ar+ + e + e (ionization, threshold 15.76 eV)
- R2: e + Ar(3p^6) -> Ar(4s[3/2]_2) + e (excitation to metastable)
- R3: e + Ar(3p^6) -> Ar(4s'[1/2]_0) + e (excitation to resonant)
- R4: e + Ar(4s[3/2]_2) -> Ar(3p^6) + e (de-excitation, detailed balancing)
- R5: e + Ar(4s'[1/2]_0) -> Ar(3p^6) + e (de-excitation, detailed balancing)
- R6: e + Ar(4s'[1/2]_0) -> Ar+ + 2e (stepwise ionization, 4.21 eV)
- R7: e + Ar(4s[3/2]_2) -> Ar+ + 2e (stepwise ionization, 4.21 eV)
- R8: e + Ar+ -> Ar2+ + 2e (double ionization, 27.63 eV)
- R9: e + Ar -> Ar2+ + 3e (direct double ionization)

**Electron-Metal (R10-R24) [Cu example]:**
- R10-R13: e + Cu(ground) -> Cu(metastable/excited) + e (excitation)
- R14-R16: Cu(metastable/excited) -> Cu(ground/other) + hv (radiative decay)
- R17-R19: e + Cu(metastable) -> Cu(other states) + e (inter-level transitions)
- R20-R23: e + Cu(various states) -> Cu+ + e (ionization from each level)
- R24: e + Cu+ -> Cu2+ + 2e (further ionization)

**Heavy particle (R25-R27):**
- R25: Ar+ + Cu -> Ar + Cu+ (charge exchange)
- R26: Ar(4s[3/2]_2) + Cu -> Ar + Cu+ + e (Penning ionization)
- R27: Ar(4s'[1/2]_0) + Cu -> Ar + Cu+ + e (Penning ionization)

Rate coefficients are parameterized as functions of T_e, stored in `data/reactions/`.

**Current limitation:** Hot-electron routing is wired into the 0D reaction assembly, but not every hot branch in the YAML package is physically populated yet. Several channels still carry zero-fit placeholders and need literature-calibrated replacement before research use.

### 3.3 Sputtering Model

From Yamamura & Tawara, sputter yield:

```
Y(E) = a * E^b
```

Where E is ion energy (eV), and for Ar+ -> Cu: a = 0.1421, b = 0.468. For Cu+ -> Cu (self-sputter): a = 0.0691, b = 0.556. Cohesive energy of Cu: 3.49 eV.

Sputtered atoms leave the target approximately in ground state Cu(3d^10 4s ^2S_{1/2}) with energy distribution peaking at roughly half the surface binding energy.

**Current implementation:** Energy-only yield Y(E) with cosine angular emission distribution for sputtered atoms. This is not an angle-dependent yield law Y(E, theta) — the cosine applies to the emission direction, not to the incidence angle dependence of the yield.

### 3.4 Key Internal Parameters (Brenning 2021)

- **alpha_t**: ionization probability of sputtered target species in the IR
- **beta_t**: back-attraction probability (fraction of ionized target species returning to target)
- **xi_t**: transport parameter (fraction of IR-escaping ion flux reaching substrate)
- **K_sput,eff**: effective sputter yield coefficient
- **epsilon_ti**: energy cost per deposited ion = e * V_HiPIMS / (K_sput,eff * xi_t * alpha_t * (1 - beta_t))

### 3.5 PIC-MCC Numerical Method

From Taccogna 2023 and Gildea 2013:

**Particle push (Boris algorithm):**
```
m_p * dv_p/dt = q_p * (E_p + v_p x B_p)
dr_p/dt = v_p
```
Leapfrog integration with constant timestep (variable timestep destabilizes leapfrog — Gildea Ch. 3). Boris rotation for magnetic field preserves |v_perp| exactly.

**Field solve:**
```
epsilon_0 * nabla^2(phi) = -e * (n_e - sum(Z_s * n_s))
E = -grad(phi)
```
Poisson equation solved on 2D(r,z) grid. **Implemented solver:** Sparse finite-difference 5-point stencil with L'Hopital treatment at r=0, solved via `cupyx.scipy.sparse.linalg.spsolve` (direct LU on GPU). Artificial permittivity factor (kappa) can relax Debye length constraint for larger cells.

**Numerical constraints:**
- CFL: v_e * dt <= dr (electron thermal velocity)
- Debye length resolution: dr <= zeta * lambda_De (zeta ~ 1, can relax with permittivity scaling)
- Collision probability: P < 5-10% per timestep to limit missed collisions
- For HiPIMS (n_e ~ 10^18 m^-3, T_e ~ 5 eV): lambda_De ~ 10^-5 m, dt ~ 10^-11 s

**Collision handling (Null-collision MCC):**
```
f(t + dt, v_p) = (1 - P) * f(t, v_p) + P * Q(v_p)
```
Nanbu no-time-counter scheme. Collision probability P based on null-collision frequency (maximum cross-section * maximum relative velocity). Accept/reject determines if collision occurs and which type.

---

## 4. Data Requirements

### Currently bundled data

| File | Contents |
|------|----------|
| `data/reactions/gudmundsson_cu_ar.yaml` | Cu/Ar rate coefficient fits k(T_e) from Gudmundsson 2022 Table 1 |
| `data/sputtering/yamamura_yields.yaml` | Yamamura Y(E) fit parameters for Ar+→Cu and Cu+→Cu |

### Required external data (not yet bundled)

| Dataset | Source | Species | Format | Status |
|---------|--------|---------|--------|--------|
| Electron-Ar elastic | Phelps / Biagi | e + Ar | sigma(E) TSV | **Bundled** in `data/cross_sections/lxcat_biagi_e_ar/` |
| Electron-Ar ionization | Phelps | e + Ar → Ar+ + 2e | sigma(E) TSV | **Bundled** in `data/cross_sections/lxcat_biagi_e_ar/` |
| Electron-Ar excitation | IST-Lisbon | e + Ar → Ar* + e | sigma(E) TSV | **Bundled** in `data/cross_sections/lxcat_biagi_e_ar/` |
| Electron-Cu ionization | Freund / IAEA | e + Cu → Cu+ + e | sigma(E) TSV | **Not bundled**; current PIC uses exploratory synthetic fallback |
| Electron-Cu excitation | Bogaerts | e + Cu → Cu* + e | sigma(E) TSV | **Not bundled**; current PIC uses exploratory synthetic fallback |
| Ar+ - Ar charge exchange | Phelps | Ar+ + Ar → Ar + Ar+ | sigma(E) TSV | **Synthetic fallback only** in current public package |
| Magnetic field maps | Analytical or measured | B_r(r,z), B_z(r,z) | .npz | **Surrogate field map bundled**; measured map still required for research-grade validation |
| Discharge waveforms | Experiment | V_D(t), I_D(t) | CSV | **Literature-fit waveform bundled** for the public Cu/Ar case |
| Ti/Ar rate coefficients | Gudmundsson | Ti/Ar reactions | YAML | **Planned, not created** |

**Download strategy (planned):** Script `scripts/download_lxcat_data.py` (not yet written) to fetch from LXCat API. All stored as two-column TSV (energy_eV, cross_section_m2).

### 4.2 Rate Coefficients

Pre-computed from Gudmundsson Table 1. Stored as parameterized fits k(T_e) in YAML. Used in 0D model directly; the PIC model computes collisions from raw cross-sections instead.

### 4.3 Magnetic Field

Static B-field computed from magnetron geometry via elliptic-integral Biot-Savart for two current loops (`magnetic.py`). Can also load a measured field map interpolated onto the grid.

### 4.4 Discharge Waveforms

The waveform loader (`data/waveforms.py`) is wired into both the PIC path and the 0D benchmark-package flow. The bundled public Cu/Ar case currently uses a literature-fit waveform rather than a measured experimental trace.

---

## 5. Codebase Structure

### Actual repo layout (as of Steps 1-8 complete)

```
plasma/
|
|-- pyproject.toml                      # uv/pip, Python >=3.11
|-- PROJECT.md                          # This file
|
|-- config/
|   |-- base.yaml                       # Shared physical constants, default numerics
|   |-- hipims_cu_ar.yaml               # Cu target, Ar gas, specific geometry
|   `-- hipims_ti_ar.yaml               # Ti target (points to missing YAML — planned)
|
|-- data/
|   |-- reactions/
|   |   `-- gudmundsson_cu_ar.yaml      # Cu/Ar rate coefficient fits from Table 1
|   |-- sputtering/
|   |   `-- yamamura_yields.yaml        # Y(E) fit parameters per ion/target pair
|   |-- cross_sections/                 # (empty — LXCat data not yet downloaded)
|   |-- magnetic_fields/                # (empty — computed analytically in code)
|   `-- waveforms/                      # (empty — no experimental waveforms yet)
|
|-- src/plasma/
|   |-- __init__.py
|   |
|   |-- core/
|   |   |-- __init__.py
|   |   |-- constants.py                # SI constants (e, m_e, epsilon_0, k_B, etc.)
|   |   |-- species.py                  # Species dataclass
|   |   |-- config.py                   # Pydantic config schema, YAML loader
|   |   `-- units.py                    # Unit conversion helpers
|   |
|   |-- data/
|   |   |-- __init__.py
|   |   |-- lxcat_parser.py             # Parse LXCat .txt / .tsv files
|   |   |-- cross_sections.py           # CrossSectionTable: interpolated sigma(E)
|   |   |-- reactions.py                # ReactionSet: load from YAML, compute rates
|   |   |-- sputtering.py              # Yamamura Y(E) model + cosine emission
|   |   `-- waveforms.py               # Waveform loader (exists, not yet wired into IRM)
|   |
|   |-- global_model/
|   |   |-- __init__.py
|   |   |-- irm.py                      # Main IRM class (reduced Cu/Ar, cold-e only)
|   |   |-- rate_equations.py           # dy/dt for species densities
|   |   |-- power_balance.py            # Electron energy equation
|   |   |-- transport.py                # Loss/gain fluxes, heuristic beta_t
|   |   `-- solver.py                   # scipy.integrate.solve_ivp wrapper
|   |
|   |-- pic/
|   |   |-- __init__.py
|   |   |-- grid.py                     # CylindricalGrid (r,z)
|   |   |-- particles.py               # ParticleArray SoA on GPU
|   |   |-- pusher.py                   # Boris pusher (Numba CUDA kernel)
|   |   |-- deposit.py                  # CIC charge deposition
|   |   |-- gather.py                   # Field interpolation: grid → particle
|   |   |-- poisson.py                  # Sparse FD Poisson solver (cupyx spsolve)
|   |   |-- boundaries.py              # Absorbing walls, axis reflection
|   |   |-- magnetic.py                # Biot-Savart B-field from current loops
|   |   |-- mcc.py                     # Hybrid CPU/GPU null-collision MCC prototype
|   |   |-- magnetron.py              # Magnetron target: simplified SEE + sputtering
|   |   |-- loop.py                    # PIC loop: deposit → solve → gather → push → BC
|   |   `-- weighting.py              # Macro-particle weight, uniform initialization
|   |
|   |-- diagnostics/__init__.py         # (stub — not yet implemented)
|   |-- io/__init__.py                  # (stub — no checkpointing/HDF5 yet)
|   `-- multi_gpu/__init__.py           # (stub — no domain decomposition yet)
|
|-- tests/
|   |-- test_cross_sections.py          # CrossSectionTable interpolation
|   |-- test_rate_coefficients.py       # k(T_e) sanity checks
|   |-- test_sputtering_yield.py        # Y(E) against Yamamura fits
|   |-- test_irm_steady_state.py        # 0D model numerical sanity checks
|   |-- test_boris_conservation.py      # Boris speed/energy conservation
|   |-- test_poisson_manufactured.py    # Method of manufactured solutions
|   |-- test_pic_integration.py         # Charge deposition, PIC step, loop
|   |-- test_mcc_rates.py              # Null-collision rates, elastic/ionization/excitation
|   `-- test_magnetron.py              # Target zone, SEE, sputtering, geometry
|
|-- scripts/
|   `-- run_global_model.py             # CLI: run 0D IRM for given config
|
`-- notebooks/                          # (empty — no notebooks yet)
```

### Planned but not yet implemented

- `diagnostics/`: EEDF, IEDF, spatial profiles, virtual probes, deposition rate
- `io/`: HDF5 checkpointing, VTK export
- `multi_gpu/`: Domain decomposition, halo exchange, NCCL particle migration
- `scripts/`: download_lxcat_data.py, generate_bfield.py, run_pic.py, benchmark_gpu.py, parameter_scan.py
- `notebooks/`: Cross-section validation, Gudmundsson reproduction, PIC benchmarks
- `config/benchmark_landau.yaml`: Landau damping PIC validation config

### Design principles

**Separation of data from physics from numerics.** The `data/` layer handles parsing and interpolation. The `global_model/` and `pic/` layers implement physics equations using data objects. The `io/` layer handles persistence. This means swapping Cu for Ti is a config change + new cross-section files, not a code change.

**GPU arrays as the default.** All particle data and field arrays live on GPU (CuPy ndarrays). CPU is used only for: config loading, ODE integration (0D model), file I/O, and orchestration. Data transfers are explicit and minimized.

**Each module has a clear input/output contract.** The Boris pusher takes (positions, velocities, E_at_particles, B_at_particles, dt, charge, mass) and returns updated (positions, velocities). No hidden state. This makes testing trivial and swapping implementations easy.

**Checkpoint everything (planned).** HDF5 snapshots at configurable intervals. The `io/` module is stubbed but not yet implemented. Currently no checkpointing or resume capability exists.

---

## 6. Dependencies

```toml
[project]
name = "plasma-sim"
requires-python = ">=3.11"
dependencies = [
    "numpy>=1.26",
    "scipy>=1.12",
    "cupy-cuda12x>=13.0",       # GPU array operations, cuFFT, cuSPARSE
    "numba>=0.59",               # CUDA kernel JIT compilation
    "pydantic>=2.5",             # Config validation
    "pyyaml>=6.0",               # Config files
    "h5py>=3.10",                # HDF5 I/O
    "matplotlib>=3.8",           # Visualization
]

[project.optional-dependencies]
dev = [
    "pytest>=8.0",
    "hypothesis>=6.98",          # Property-based testing (conservation laws)
    "ruff>=0.3",                 # Linting
]
cluster = [
    "mpi4py>=3.1",               # Multi-GPU communication
    "cupy-cuda12x[nccl]",        # NCCL for GPU-GPU transfers
]
ml = [
    "torch>=2.2",                # Surrogate models, neural operators
    "wandb>=0.16",               # Experiment tracking
]
viz = [
    "pyvista>=0.43",             # 3D field visualization
    "ipywidgets>=8.1",           # Interactive notebooks
]
```

### Why these choices

| Dependency | Why | Alternative considered |
|------------|-----|----------------------|
| **CuPy** | Drop-in NumPy on GPU, includes cuFFT for Poisson solver, cuSPARSE for sparse matrices. Mature, well-documented. | PyTorch (heavier, overkill for array ops), Taichi (less ecosystem) |
| **Numba CUDA** | Write GPU kernels in Python. Boris pusher and charge deposition are simple enough that Numba gets ~90% of hand-written CUDA performance. | Raw CUDA C (faster iteration in Python), CuPy RawKernels (less ergonomic) |
| **scipy** | solve_ivp for 0D ODE system. Battle-tested stiff solvers (BDF, Radau) needed for the rate equations. | Hand-rolled RK4 (insufficient for stiff chemistry) |
| **Pydantic** | Config validation with type checking. Catches bad parameters before a 10-hour run starts. | dataclasses (no validation), attrs (less ecosystem) |
| **HDF5** | Standard in computational physics. Parallel I/O support for multi-GPU. Handles large arrays efficiently. | NetCDF4 (similar but less common in plasma), Zarr (better for cloud, worse for HPC) |

---

## 7. Extensibility Roadmap

The architecture is designed so that each future extension is an **additive module**, not a rewrite.

### Phase 1: Core Simulation (current scope)
- 0D global model: reduced Cu/Ar IRM prototype with numerical sanity-check tests (not yet reproducing specific Gudmundsson 2022 figures/tables)
- 2D PIC-MCC on single GPU: numerical core implemented (Boris, Poisson, CIC, MCC, magnetron boundaries), module-level tests passing
- **Not yet done:** benchmark reproduction (Landau damping, sheath scaling, IEDF), Ti/Ar chemistry, diagnostics, full HiPIMS pulse integration

### Phase 2: ML Surrogates
- Train neural operator (FNO or DeepONet) on PIC output fields
- Input: process parameters (V, p, B, t_pulse) -> Output: n_e(r,z), IEDF, alpha_t, beta_t
- 1000x faster than PIC for parameter exploration
- **New module:** `src/plasma/ml/surrogate.py`, `src/plasma/ml/training.py`
- **Data pipeline:** PIC checkpoints -> training dataset (HDF5 -> PyTorch DataLoader)

### Phase 3: Computer Vision Bridge (Plasma Orbs / OES)
- Capture video of plasma globe or OES spectra from real discharge
- Extract spatial intensity maps, filament dynamics, spectral line ratios
- Compare to simulated emission (from excited state densities in simulation)
- **New module:** `src/plasma/cv/capture.py`, `src/plasma/cv/features.py`
- **Bridge:** simulated OES from `diagnostics/` <-> measured OES from `cv/`
- This creates a **sim-to-real loop**: simulation predicts emission, camera measures it, discrepancy trains a correction model

### Phase 4: Closed-Loop Control
- ML surrogate predicts plasma state from process parameters
- CV module observes real plasma in real-time
- Controller adjusts pulse parameters (V, t_pulse, frequency) to hit target film properties
- **New module:** `src/plasma/control/controller.py`, `src/plasma/control/optimizer.py`
- Requires: fast surrogate (<100ms inference), real-time camera feed, hardware interface to pulse generator

### What makes this extensible

```
                    +-----------+
                    |  CONFIG   |   <- YAML files define everything
                    +-----+-----+
                          |
          +---------------+---------------+
          |               |               |
    +-----v-----+  +-----v-----+  +------v----+
    | DATA LAYER |  |  0D MODEL |  |  PIC-MCC  |
    | (cross sec,|  | (fast,    |  | (accurate,|
    |  reactions)|  |  seconds) |  |  hours)   |
    +-----+-----+  +-----+-----+  +-----+-----+
          |               |               |
          +-------+-------+-------+-------+
                  |               |
           +------v------+ +-----v------+
           | DIAGNOSTICS | |   IO/HDF5  |
           | (EEDF, IEDF | | checkpoint |
           |  profiles)  | +-----+------+
           +------+------+       |
                  |               |
          +-------v-------+------v------+
          |  ML SURROGATE | TRAINING    |  <- Phase 2: learns from PIC data
          |  (fast proxy) | PIPELINE    |
          +-------+-------+------+------+
                  |               |
          +-------v-------+------v------+
          |   CV MODULE   | SIM-TO-REAL |  <- Phase 3: camera + comparison
          |  (camera,     | BRIDGE      |
          |   features)   |             |
          +-------+-------+------+------+
                  |               |
          +-------v---------------v-----+
          |     CLOSED-LOOP CONTROL     |  <- Phase 4: real-time optimization
          +-----------------------------+
```

Each layer only depends on the layer above it. The ML surrogate consumes HDF5 files from the PIC; it doesn't import PIC code. The CV module compares features against diagnostic outputs; it doesn't need to know how those diagnostics were computed. This means any layer can be developed, tested, and improved independently.

---

## 8. Validation

### Completed numerical verification (module-level tests)

| Test suite | What it verifies | # tests |
|------------|-----------------|---------|
| test_cross_sections | CrossSectionTable interpolation, log-log, zero-below-threshold | 6 |
| test_rate_coefficients | k(T_e) order-of-magnitude sanity, Arrhenius shape | 9 |
| test_sputtering_yield | Y(E) against Yamamura fits, threshold, Cu/Ti targets | 8 |
| test_irm_steady_state | 0D model converges, energy conservation, density positivity | 13 |
| test_boris_conservation | Speed conservation in B, KE conservation, cyclotron period, axis reflection | 5 |
| test_poisson_manufactured | MMS quadratic solution, linear potential → constant E, zero-rho → zero-phi | 5 |
| test_pic_integration | Charge deposition, quasineutrality, PIC step/loop execution | 8 |
| test_mcc_rates | Null-collision probability scaling, elastic/ionization/excitation physics | 10 |
| test_magnetron | Target zone detection, absorption, SEE yields, sputtering, B-field | 10 |

### Planned validation targets (not yet implemented)

These require end-to-end simulation runs and diagnostics that do not yet exist:

| Milestone | Target | Source | Pass criterion |
|-----------|--------|--------|---------------|
| Cross-section data | sigma_ionization(Ar) at 100 eV | Phelps database | Within 5% of tabulated value |
| 0D model: Cu/Ar Case I | n_e(t), T_e(t) during 40 us pulse | Gudmundsson 2022 Fig. 3a | Qualitative match of temporal evolution |
| 0D model: Cu/Ar Case II | alpha_t = 61-69%, beta_t = 44-50% | Gudmundsson 2022 Table 4 | Within reported range |
| 0D model: optimization | epsilon_ti vs V_HiPIMS trend | Brenning 2021 Fig. 4 | Correct monotonic trend |
| PIC: Landau damping | Damping rate gamma | Analytical (Birdsall & Langdon) | Within 5% of analytical |
| PIC: magnetron sheath | Sheath thickness vs n_e | Child-Langmuir law | Correct scaling |
| PIC: HiPIMS IEDF | Cu+ energy distribution at substrate | Cemin 2016 / Gudmundsson 2022 | Peak position within 20% |

---

## 9. Build Order

1. **Project scaffold + config system** — pyproject.toml, pydantic configs, constants
2. **Cross-section data pipeline** — LXCat parser, interpolation, validation notebook
3. **0D Global Model** — Reproduce Gudmundsson 2022 for Cu/Ar
4. **Brenning optimization** — Implement flow chart analysis from Brenning 2021
5. **PIC core** — Grid, particles, Boris pusher, Poisson solver, charge deposition
6. **PIC validation** — Landau damping benchmark
7. **MCC collisions** — Null-collision method with LXCat cross sections
8. **Magnetron geometry** — B-field, sputtering boundaries, SEE
9. **Full HiPIMS PIC** — Complete pulse simulation on 4080
10. **Multi-GPU** — Domain decomposition for H100 cluster
11. **Production runs** — Parameter scans, IEDF extraction, publication figures

---

## 10. References

- Gudmundsson, J.T. et al. "Ionization region model of high power impulse magnetron sputtering of copper." Surface & Coatings Technology 442 (2022) 128189.
- Brenning, N. et al. "HiPIMS optimization by using mixed high-power and low-power pulsing." Plasma Sources Sci. Technol. 30 (2021) 015015.
- Taccogna, F. et al. "Plasma propulsion modeling with particle-based algorithms." J. Appl. Phys. 134 (2023) 150901.
- Gildea, S.R. "Development of the Plasma Thruster Particle-in-Cell Simulator..." MIT PhD Thesis (2013).
- Hsu, T.-W. "Effect of metal ion irradiation on hard coating synthesis by PVD." Linkoping University Dissertation No. 2288 (2023).
