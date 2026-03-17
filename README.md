# Plasma Sputtering Simulator

GPU-accelerated plasma simulation for **High Power Impulse Magnetron Sputtering (HiPIMS)**, targeting thin-film deposition process understanding and optimization.

## What It Does

Two complementary simulation fidelities for HiPIMS discharges:

- **0D Global Model (IRM)** — Volume-averaged ODE system tracking species densities and electron temperature. Runs in seconds. Based on the Gudmundsson 2022 ionization region model for Cu/Ar.

- **2D Axisymmetric PIC-MCC** — Fully kinetic Particle-in-Cell with Monte Carlo Collisions in cylindrical (r, z) geometry. Self-consistent electric fields via Poisson solver, prescribed static magnetic field from magnetron geometry. Runs on NVIDIA GPU via CuPy/Numba CUDA.

## Features

- Full Cu/Ar reaction set (27 reactions from Gudmundsson 2022)
- Boris pusher with exact magnetic rotation
- Sparse finite-difference Poisson solver on GPU (`cupyx.scipy.sparse.linalg.spsolve`)
- Null-collision Monte Carlo for electron-neutral and ion-neutral interactions
- Yamamura sputtering yield model with cosine angular emission
- Magnetron geometry with secondary electron emission
- Pydantic-validated configuration via YAML files
- HDF5 checkpoint/restart support
- Real-time viewer (PySide6/pyqtgraph) for live monitoring of PIC runs
- Docker support with NVIDIA GPU passthrough

## Quick Start

### Requirements

- Python >= 3.11
- NVIDIA GPU with CUDA 12.x
- [uv](https://github.com/astral-sh/uv) (recommended) or pip

### Install

```bash
git clone https://github.com/kyle-compute/plasma.git
cd plasma
uv venv && uv pip install -e ".[dev]"
```

### Run the 0D Global Model

```bash
python scripts/run_global_model.py --config config/hipims_cu_ar.yaml
```

### Run PIC Simulation

```bash
plasma-tools pic --config config/hipims_cu_ar_pic.yaml
```

### Docker

```bash
docker compose up --build
```

Requires [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html) for GPU passthrough.

## Project Structure

```
plasma/
├── config/                 # YAML configuration files
├── data/
│   ├── reactions/          # Rate coefficient fits (Gudmundsson 2022)
│   ├── sputtering/         # Yamamura yield parameters
│   ├── cross_sections/     # LXCat cross-section data
│   ├── collision_packages/ # Bundled collision data packages
│   └── material_packages/  # Material property packages
├── src/plasma/
│   ├── core/               # Constants, config, species definitions
│   ├── data/               # Cross-sections, reactions, sputtering models
│   ├── global_model/       # 0D IRM (rate equations, power balance, transport)
│   ├── pic/                # 2D PIC-MCC (Boris, Poisson, MCC, magnetron)
│   ├── diagnostics/        # IEDF, spatial profiles, distribution functions
│   ├── io/                 # HDF5 checkpointing
│   ├── live/               # Real-time data publishing for viewer
│   ├── viewer/             # PySide6 live monitoring application
│   └── ml/                 # ML surrogate models (experimental)
├── tests/                  # pytest suite
├── scripts/                # CLI entry points
├── Dockerfile              # GPU-enabled container
└── docker-compose.yml
```

## Configuration

All simulation parameters are defined in YAML config files. See `config/base.yaml` for defaults and `config/hipims_cu_ar.yaml` for a complete Cu/Ar example.

Key parameters:
- Discharge voltage, pressure, pulse timing
- Target material (Cu, Ti) and gas (Ar)
- Grid resolution, timestep, particle count
- Diagnostics intervals and output paths

## Tests

```bash
pytest
```

The test suite covers cross-section interpolation, rate coefficients, sputtering yields, Boris conservation, Poisson solver (method of manufactured solutions), charge deposition, MCC collision rates, and magnetron geometry.

## Physics References

- Gudmundsson, J.T. et al. "Ionization region model of high power impulse magnetron sputtering of copper." *Surf. Coat. Technol.* 442 (2022) 128189.
- Brenning, N. et al. "HiPIMS optimization by using mixed high-power and low-power pulsing." *Plasma Sources Sci. Technol.* 30 (2021) 015015.
- Taccogna, F. et al. "Plasma propulsion modeling with particle-based algorithms." *J. Appl. Phys.* 134 (2023) 150901.

## License

MIT
