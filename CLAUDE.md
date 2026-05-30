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
uv run python convergence_study.py               # dt-convergence study + plot (kinetic-temperature bias ~ dt^2)
uv run python manage_artifacts.py list|clean [--delete]
uv run --extra dev pytest                        # full test suite
uv run --extra dev pytest -m "not statistical"   # fast deterministic tests only (~2s)
uv run --extra dev ruff check .                  # lint
```

## Testing

`tests/` is a `pytest` suite (`uv run --extra dev pytest`):
- **Deterministic** (`test_potential.py`, `test_integrator.py`, `test_io.py`): critical-point Hessian signatures, analytical-vs-autograd force, NVE energy conservation at friction=0 (BAOAB → velocity-Verlet), seed reproducibility, HDF5 round-trip, shape/dtype invariants. Fast, hard assertions.
- **Statistical** (`test_sampling.py`, marked `@pytest.mark.statistical`): equipartition, harmonic distributions, the Müller-Brown Boltzmann ⟨V⟩ vs a grid-integrated canonical average, free-particle VACF, and the dt² kinetic-temperature convergence order. Seeded with tolerances above the sampling error. Exclude with `-m "not statistical"`.

Key physics gotcha the tests encode: BAOAB samples *configurational* averages accurately (exactly for a harmonic oscillator) but the *kinetic* temperature carries an O(dt²) bias that grows with curvature — ~4% on Müller-Brown at dt=0.01. So equipartition is tested on soft/free systems, not Müller-Brown; `convergence_study.py` characterizes the bias vanishing as dt→0.

`verify_langevin.py` remains as a standalone script (BAOAB position/velocity variances vs the analytic harmonic distribution, exits non-zero on failure).

## Architecture

Data flows: `MuellerBrownPotential` (energy + forces) → `LangevinSimulator` (trajectory) → results dict → `io.py` (HDF5) and/or `MuellerBrownVisualizer` (plots). `main.py` orchestrates via its `--mode` CLI; `config.py` builds the standardized config dict that threads through every mode.

Things that take reading multiple files to see:

- **Forces have two implementations** (`potential.py`), selected by `use_autograd`: the hand-derived analytical gradient (default, faster) and `torch.autograd.grad` (reference). They exist to cross-check each other — keep them numerically consistent. The hot paths (`_calculate_potential`, `_calculate_force`) are standalone `@torch.jit.script` functions; the module just dispatches to them. Potential parameters are registered buffers, so device/dtype move with the module.
- **`simulate(..., observables=...)`** stores and returns only the requested trajectories (positions always kept) plus metadata in one dict; trajectory arrays are `(n_save_steps, n_particles, 2)`. Forces are always computed to drive the dynamics but only stored if requested; energy is only computed when stored. Defaults to all four when `observables=None`. `config.py`'s observable list flows straight through to this argument.
- **`config.py`** produces the nested `simulation`/`initial_conditions`/`output` dict consumed everywhere; `create_experiment_config()` is the single source of defaults.

## Conventions
- **Imports are absolute from the repo root** (`from src.muller_brown...`), so run all scripts from the project root, not from inside `src/`.
- **Default dtype is `torch.float64`, device CPU** (`constants.py`). Keep simulation code in double precision; scientific correctness and reproducibility come first.
- Canonical potential parameters and critical-point locations live in `constants.py` / `potential.py` — reuse them, don't re-hardcode.
- Type hints: builtin generics and `|` only — `list`/`dict`/`tuple`/`set`, `X | None` (not `Optional`), `A | B` (not `Union`). `_` prefix for private members, `ALL_CAPS` for constants.
- Comments only where they clarify the non-obvious. Favor concision and low complexity; simplify where possible, never at the cost of correctness.
