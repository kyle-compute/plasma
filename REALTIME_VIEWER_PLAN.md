# Realtime Visualization Plan

## Goal

Add a realtime, controllable visualization layer for the 0D and PIC solvers without destabilizing the physics code or forcing the project into a second low-level GPU stack.

This plan aims for:

- live viewing while the sim runs
- pause/resume/step controls
- live diagnostics and field maps
- saved snapshots, movies, and reports
- a path to richer rendering later

It does not assume literal wall-clock plasma evolution. The target is interactive scientific monitoring, not game-engine-speed full-fidelity HiPIMS.

## Recommendation

Do not start with Vulkan.

Best near-term stack:

- `PySide6` for the app shell and controls
- `pyvistaqt` / `PyVista` for mesh and field visualization
- `pyqtgraph` for high-rate timeseries panels
- a separate simulation process
- a local streaming channel between sim and viewer

Why:

- the simulation is already CUDA/CuPy/Numba-based
- VTK/PyVista fits scientific grids and scalar fields well
- Qt gives controllable desktop UI quickly
- this is much lower risk than building a custom Vulkan renderer
- it keeps the rendering layer replaceable later

Vulkan or WebGPU should only be considered after the first viewer exists and measurement shows the viewer is the bottleneck.

## Product Shape

The system should have two runtime modes:

1. `Live scientific viewer`
   - attach to a running 0D or PIC case
   - watch diagnostics and reduced field snapshots update live
   - pause, resume, single-step, checkpoint, and save views

2. `Quick interactive mode`
   - lower-resolution PIC or reduced-order surrogate
   - responds quickly to parameter changes
   - used for exploration, not publication-grade validation

The full-fidelity public PIC case remains a slower backend path.

## Architecture

Use a split-process design.

### Simulation Process

Responsibilities:

- run the solver
- publish snapshots every `N` timesteps
- expose a control API
- write checkpoints and final outputs

It should never depend on Qt or UI code.

### Viewer Process

Responsibilities:

- subscribe to snapshots
- render live fields, particles, and diagnostics
- send commands back to the sim
- record screenshots or animation frames

It should never own solver logic.

### Control Plane

Supported commands:

- `pause`
- `resume`
- `single_step`
- `stop`
- `save_checkpoint`
- `set_diag_interval`
- `set_render_interval`
- `set_particle_subsample`

Only safe runtime-mutated parameters should be changeable during a run.

## Data Flow

Use coarse-grained snapshot publishing, not per-step rendering.

Recommended default cadence:

- publish every `100-1000` PIC steps
- render at `5-20 FPS`
- always decimate before shipping to the viewer

Each snapshot should contain:

- timestamp
- step number
- case name
- provenance flags
- selected scalar fields
- selected particle sample
- live diagnostics
- optional histogram data

## Snapshot Contract

Add a typed contract for live updates.

Suggested modules:

- `src/plasma/live/contracts.py`
- `src/plasma/live/publisher.py`
- `src/plasma/live/subscriber.py`
- `src/plasma/live/control.py`

Suggested snapshot models:

- `LiveRunState`
- `FieldSnapshot2D`
- `ParticleSample`
- `TimeseriesChunk`
- `HistogramSnapshot`
- `ViewerCommand`

### Minimum PIC Snapshot Payload

- `phi`
- `rho`
- `|E|`
- `Br`, `Bz` or `|B|`
- species particle counts
- energy totals
- target impacts / sputter / SEE counters
- substrate IEDF histogram

### Minimum 0D Snapshot Payload

- `n_e`
- `T_e`
- `current`
- `voltage`
- `alpha_t`
- `beta_t`
- `xi_t`
- deposition flux

## Viewer Layout

Phase 1 desktop layout:

- top-left: live field map
- top-right: live timeseries dashboard
- bottom-left: particle view or phase-space slice
- bottom-right: substrate IEDF / collision activity / status panel

Controls:

- run / pause / step
- field selector
- species selector
- particle subsample slider
- snapshot cadence
- save frame / save movie / save checkpoint

## Rendering Strategy

### Phase 1

Use PyVista for:

- 2D scalar field heatmaps
- contour overlays
- sampled particle point clouds

Use pyqtgraph for:

- current and voltage traces
- species count traces
- energy traces
- event counters

### Phase 2

Add:

- field animation export
- side-by-side comparison overlays
- provenance and validation badges
- uncertainty / heuristic markers

### Phase 3

Evaluate faster render backends only if needed:

- `pygfx` / `wgpu` before raw Vulkan
- raw Vulkan only if:
  - VTK becomes the bottleneck
  - zero-copy GPU interop is required
  - particle counts exceed what PyVista can handle interactively

## Performance Rules

- never stream full particle arrays every frame
- always subsample particles for viewing
- downsample fields for live display
- keep full-resolution data for checkpoints and offline analysis
- separate simulation cadence from render cadence
- treat the viewer as lossy by design

Initial target:

- sim runs as fast as possible
- viewer updates at `10 FPS`
- UI commands feel responsive within `250 ms`

## Repo Changes

Suggested additions:

```text
src/plasma/live/
  __init__.py
  contracts.py
  publisher.py
  subscriber.py
  control.py

src/plasma/viewer/
  __init__.py
  app.py
  main_window.py
  field_panel.py
  particle_panel.py
  diagnostics_panel.py
  controls_panel.py

scripts/
  run_live_viewer.py
  run_pic_live.py
  run_global_live.py
```

Suggested config additions:

- `live.enabled`
- `live.publish_interval`
- `live.transport`
- `live.max_particles_view`
- `live.fields`

## Transport Choice

Start simple:

- local `ZeroMQ` or local TCP websocket

Do not start with shared CUDA memory interop.

Reason:

- easier to debug
- process separation stays clean
- plenty fast for decimated scientific snapshots

If needed later:

- shared memory for CPU snapshots
- GPU-aware zero-copy only after measurement justifies it

## Implementation Phases

### Phase 1: Live Diagnostics Backbone

- add typed snapshot and command contracts
- add publisher hooks to the 0D runner
- add publisher hooks to the PIC loop
- emit reduced snapshots on a configurable cadence
- add a minimal CLI viewer that prints state and receives commands

Acceptance:

- a running sim publishes snapshots and responds to pause/resume

### Phase 2: Desktop Viewer

- build Qt shell
- add PyVista field panel
- add pyqtgraph diagnostics panel
- add controls for pause/resume/step/checkpoint
- support both 0D and PIC sources

Acceptance:

- user can watch the sim update live and control execution

### Phase 3: Particle and Histogram Views

- add particle subsampling and rendering
- add substrate IEDF and collision histograms
- add camera presets and species filters

Acceptance:

- viewer shows both fields and particles without dragging down sim throughput badly

### Phase 4: Recording and Reporting

- save frame sequences
- export mp4 or gif
- save synchronized report bundles
- add provenance labels onto visuals

Acceptance:

- one run can produce both live viewing and reusable report artifacts

### Phase 5: Quick Interactive PIC Mode

- add a reduced-size `quicklook` PIC config
- expose a small set of safely mutable parameters
- make restart and stepping reliable

Acceptance:

- user can run an interactive plasma demo in minutes, not hours

### Phase 6: Advanced Renderer Evaluation

- benchmark current viewer performance
- only if needed, prototype `pygfx/wgpu`
- compare against PyVista before committing

Acceptance:

- renderer choice is evidence-based, not speculative

## Testing

- contract tests for snapshot serialization
- integration tests for publisher/subscriber message flow
- control-plane tests for pause/resume/step/checkpoint
- viewer smoke tests for startup and panel updates
- performance tests on snapshot rate and UI latency

## Risks

- trying to render too much data live will make the viewer useless
- mutating unsafe solver parameters mid-run can invalidate physics
- adding UI code directly into solver modules will create a maintenance mess
- switching to Vulkan too early will consume effort without improving scientific value

## Best Next Steps

1. Add live snapshot contracts and publisher hooks.
2. Add a minimal viewer shell before any fancy rendering.
3. Add a `quicklook` PIC config specifically for live interaction.
4. Add provenance overlays so live visuals remain scientifically honest.

## Definition of Done

This feature is complete when:

- a user can launch a live PIC or 0D run
- a separate viewer updates while the run is active
- the user can pause, resume, step, and checkpoint
- live visuals include fields, diagnostics, and particle samples
- saved outputs still match the existing report/manifest structure
- the full-fidelity sim remains decoupled from viewer code
