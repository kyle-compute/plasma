# Plasma Sputtering Simulator

A GPU-accelerated HiPIMS (High Power Impulse Magnetron Sputtering) plasma simulation. Built as a learning project and engineering baseline — not a validated research tool. The full PIC-MCC pipeline runs end-to-end on an NVIDIA GPU, but the collision cross-sections are synthetic approximations and the simulation hasn't been benchmarked against published experimental data yet.

If you're looking for a starting point to build a real HiPIMS simulator on top of, this might be useful. If you need production-grade plasma modeling, look at VSIM, WARP, or PICCOLO.

## What's here

**0D Global Model (IRM)** — Volume-averaged rate equations for species densities and electron temperature in the ionization region. Uses Cu/Ar reaction rate fits from Gudmundsson 2022 (real literature data). Runs in seconds. This is the more trustworthy part of the code.

**2D Axisymmetric PIC-MCC** — Particle-in-Cell with Monte Carlo Collisions in cylindrical (r, z) geometry. Boris pusher, sparse Poisson solver, null-collision MCC, magnetron boundaries with sputtering and SEE. The numerical machinery works, but the physics inputs (cross-sections) are synthetic stand-ins, not measured data.

## What works

- Full PIC loop: deposit → Poisson solve → gather → Boris push → boundaries → MCC → diagnostics
- Checkpoint/restart via HDF5 (can survive long runs)
- IEDF and EEDF extraction, spatial density/temperature profiles
- Real-time viewer for watching PIC runs live (PySide6/pyqtgraph)
- 0D model with literature-sourced Cu/Ar rate coefficients (27 reactions)
- Docker container with GPU passthrough

## What doesn't work (yet)

- **Cross-sections are synthetic.** The e-Ar and e-Cu collision cross-sections are analytical approximations, not real LXCat/Phelps data. The plumbing to load real data exists, the data hasn't been plugged in.
- **No multi-GPU.** Single GPU only, no domain decomposition. Each GPU runs an independent simulation.
- **Poisson solver won't scale.** Uses sparse direct LU — fine for the current 60x100 grid, will choke on anything above ~150x150. Needs a multigrid or iterative solver for larger domains.
- **Simplified surface physics.** SEE yield is constant (not energy/angle-dependent). Sputtering yield is energy-only (no incidence angle dependence).
- **Not validated.** No benchmark reproduction (Landau damping, Gudmundsson figures, experimental IEDFs). The test suite covers module-level correctness, not physics accuracy.
- **Permittivity inflation.** Uses kappa=10 to relax Debye length constraints, which distorts sheath structure.

## Quick start

Requires Python >= 3.11 and an NVIDIA GPU with CUDA 12.x.

```bash
git clone https://github.com/kyle-compute/plasma.git
cd plasma
uv venv && uv pip install -e ".[dev]"
```

Run the 0D model:
```bash
python scripts/run_global_model.py --config config/hipims_cu_ar.yaml
```

Run PIC:
```bash
plasma-tools pic --config config/hipims_cu_ar_pic.yaml
```

Docker:
```bash
docker compose up --build   # needs NVIDIA Container Toolkit
```

Tests:
```bash
pytest
```

## Project structure

```
plasma/
├── config/                 # YAML configs (discharge params, grid, timing)
├── data/
│   ├── reactions/          # Rate coefficient fits from Gudmundsson 2022
│   ├── sputtering/         # Yamamura yield parameters
│   ├── collision_packages/ # Bundled collision data (synthetic)
│   └── material_packages/  # Material property packages
├── src/plasma/
│   ├── core/               # Constants, config, species
│   ├── data/               # Cross-sections, reactions, sputtering
│   ├── global_model/       # 0D IRM
│   ├── pic/                # 2D PIC-MCC (Boris, Poisson, MCC, magnetron)
│   ├── diagnostics/        # IEDF, EEDF, spatial profiles
│   ├── io/                 # HDF5 checkpointing
│   ├── viewer/             # Live PIC monitoring (PySide6)
│   └── ml/                 # ML surrogate models (experimental)
├── tests/                  # pytest suite
├── Dockerfile
└── docker-compose.yml
```

## If you want to make this useful

The single highest-impact improvement is replacing the synthetic cross-sections with real data. Download e-Ar cross-sections from [LXCat](https://lxcat.net/) (Biagi or Phelps sets), format them as two-column TSV, and point the collision package config at them. The `CollisionPackage` loader and `lxcat_import` module already handle the rest.

After that: bump the grid to 128x200+, increase run duration to cover a full HiPIMS pulse (~100 µs), and run a Landau damping benchmark to verify the PIC machinery independently of the collision physics.

## References

- Gudmundsson, J.T. et al. "Ionization region model of high power impulse magnetron sputtering of copper." *Surf. Coat. Technol.* 442 (2022) 128189.
- Brenning, N. et al. "HiPIMS optimization by using mixed high-power and low-power pulsing." *Plasma Sources Sci. Technol.* 30 (2021) 015015.
- Taccogna, F. et al. "Plasma propulsion modeling with particle-based algorithms." *J. Appl. Phys.* 134 (2023) 150901.

## License

MIT
