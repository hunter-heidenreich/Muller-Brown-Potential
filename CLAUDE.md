# CLAUDE.md

## Overview

PyTorch implementation of the Müller-Brown 2D potential energy surface with Langevin dynamics simulation. Used for studying rare-event transitions between minima. The package lives under `src/muller_brown/`; top-level scripts (`main.py`, `example.py`, `benchmark.py`, `nnp.py`, `manage_artifacts.py`) are entry points.

## Commands

This project uses `uv`. Run everything through it.

```bash
uv sync                                          # install dependencies
uv run python main.py --mode demo                # show potential info (minima, saddles, sample energies/forces)
uv run python main.py --mode single --save-plots # one simulation -> new artifacts/<timestamp>/ dir
uv run python main.py --mode batch --n-simulations 10 --save-plots
uv run python main.py --mode plot --artifact-dir artifacts/<timestamp>  # regenerate plots from saved data
uv run python main.py --mode list                # list artifact directories
uv run python example.py                         # minimal programmatic example
uv run python benchmark.py                       # analytical-vs-autograd force timing
uv run python verify_langevin.py                 # validate BAOAB integrator vs analytic harmonic oscillator
uv run python manage_artifacts.py list|clean [--delete]
uv run ruff check .                              # lint (ruff is the only dev dependency)
```

There is **no unit-test suite**. `verify_langevin.py` is the closest thing to a correctness test — it samples a harmonic oscillator with the BAOAB integrator and checks the position/velocity variances against the analytic canonical distribution (exits non-zero on failure). Other verification is by running simulations and inspecting output/plots.

## Architecture

Data flows: `MuellerBrownPotential` (energy + forces) → `LangevinSimulator` (trajectory) → results dict → `io.py` (HDF5) and/or `MuellerBrownVisualizer` (plots). `main.py` orchestrates via its `--mode` CLI; `config.py` builds the standardized config dict that threads through every mode.

Things that take reading multiple files to see:

- **Forces have two implementations** (`potential.py`), selected by `use_autograd`: the hand-derived analytical gradient (default, faster) and `torch.autograd.grad` (reference). They exist to cross-check each other — keep them numerically consistent. The hot paths (`_calculate_potential`, `_calculate_force`) are standalone `@torch.jit.script` functions; the module just dispatches to them. Potential parameters are registered buffers, so device/dtype move with the module.
- **`simulate()` always computes all four observables** (positions, velocities, forces, potential_energy) and returns them plus metadata in one dict; trajectory arrays are `(n_save_steps, n_particles, 2)`. `config.py`'s observable list only controls what gets *saved/plotted* downstream, not what's computed.
- **`config.py`** produces the nested `simulation`/`initial_conditions`/`output` dict consumed everywhere; `create_experiment_config()` is the single source of defaults.

## Conventions
- **Imports are absolute from the repo root** (`from src.muller_brown...`), so run all scripts from the project root, not from inside `src/`.
- **Default dtype is `torch.float64`, device CPU** (`constants.py`). Keep simulation code in double precision; scientific correctness and reproducibility come first.
- Canonical potential parameters and critical-point locations live in `constants.py` / `potential.py` — reuse them, don't re-hardcode.
- Type hints: builtin generics and `|` only — `list`/`dict`/`tuple`/`set`, `X | None` (not `Optional`), `A | B` (not `Union`). `_` prefix for private members, `ALL_CAPS` for constants.
- Comments only where they clarify the non-obvious. Favor concision and low complexity; simplify where possible, never at the cost of correctness.
