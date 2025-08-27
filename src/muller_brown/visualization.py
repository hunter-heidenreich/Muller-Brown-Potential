"""Modern visualization tools for the Müller-Brown potential and simulation data."""

import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from pathlib import Path

from src.muller_brown.potential import MuellerBrownPotential
from src.muller_brown.data import convert_to_tensor


# Default plotting constants
DEFAULT_FIGURE_SIZE = (12, 4)
DEFAULT_X_RANGE = (-2.0, 2.0)
DEFAULT_Y_RANGE = (-1.0, 3.0)
DEFAULT_RESOLUTION = 200
DEFAULT_BATCH_SIZE = 10000
DEFAULT_CMAP = "viridis"
DEFAULT_LEVELS = 24


class MuellerBrownVisualizer:
    """Modern visualization tools for Müller-Brown potential and simulation observables."""

    def __init__(self, potential: MuellerBrownPotential):
        """Initialize the visualizer."""
        self.potential = potential

    def plot_potential_surface(
        self,
        x_range: tuple[float, float] = DEFAULT_X_RANGE,
        y_range: tuple[float, float] = DEFAULT_Y_RANGE,
        resolution: int = DEFAULT_RESOLUTION,
        ax: Axes | None = None,
        cmap: str = DEFAULT_CMAP,
        levels: int = DEFAULT_LEVELS,
    ) -> tuple[Figure, Axes]:
        """Plot the Müller-Brown potential energy surface with critical points."""
        # Input validation
        if resolution <= 0:
            raise ValueError(f"resolution must be positive, got {resolution}")
        if x_range[0] >= x_range[1]:
            raise ValueError(f"Invalid x_range: {x_range}")
        if y_range[0] >= y_range[1]:
            raise ValueError(f"Invalid y_range: {y_range}")
        if levels <= 0:
            raise ValueError(f"levels must be positive, got {levels}")

        # Create coordinate grid
        x = np.linspace(x_range[0], x_range[1], resolution)
        y = np.linspace(y_range[0], y_range[1], resolution)
        X, Y = np.meshgrid(x, y)

        # Compute potential on grid in batches
        V = self._compute_potential_grid(X, Y)

        # Apply log transformation for better visualization
        V_log = self._apply_log_transform(V)

        # Create plot
        fig, ax = self._setup_figure_and_axes(ax, (10, 8))

        # Plot contours with improved styling
        v_min, v_max = V_log.min(), V_log.max()
        contour_levels = np.linspace(v_min, v_max, levels)

        contour = ax.contourf(
            X, Y, V_log, levels=contour_levels, cmap=cmap, extend="both"
        )

        # Add contour lines for better definition
        ax.contour(
            X, Y, V_log, levels=contour_levels[::3], colors='black', alpha=0.3, linewidths=0.5
        )

        # Add colorbar with improved styling
        cbar = fig.colorbar(contour, ax=ax, shrink=0.8, aspect=30)
        cbar.set_label(r"$\log(V - V_{\min})$", rotation=270, labelpad=20, fontsize=12)
        cbar.ax.tick_params(labelsize=10)

        # Add critical points
        self._add_critical_points(ax)

        # Labels and title with improved styling
        ax.set_xlabel(r"$x$", fontsize=14, fontweight='bold')
        ax.set_ylabel(r"$y$", fontsize=14, fontweight='bold')
        ax.set_title("Müller-Brown Potential Energy Surface", fontsize=16, fontweight='bold', pad=20)
        
        # Improve grid and axis styling
        ax.grid(True, alpha=0.2, linestyle='--', linewidth=0.5)
        ax.tick_params(labelsize=11)
        
        # Set aspect ratio to be equal for better visualization
        ax.set_aspect('equal', adjustable='box')

        return fig, ax

    def plot_position_distributions(
        self, data: dict, sample_idx: int | None = None
    ) -> tuple[Figure, np.ndarray]:
        """Plot position distributions: 1D marginals and 2D joint distribution.
        
        Args:
            data: Simulation data dictionary
            sample_idx: If provided, plot only for this particle index. If None (default),
                       aggregate across all particles.
                       
        Raises:
            ValueError: If positions data is not available
        """
        if "positions" not in data:
            raise ValueError("Position data not available - was 'positions' included in saved observables?")
            
        positions_raw = data["positions"]  # (n_steps, n_particles, 2)
        
        if sample_idx is not None:
            # Single particle mode
            positions = positions_raw[:, sample_idx, :]  # (n_steps, 2)
            title_suffix = f" (Particle {sample_idx})"
        else:
            # All particles mode - flatten across particles and time steps
            positions = positions_raw.reshape(-1, 2)  # (n_steps * n_particles, 2)
            n_particles = positions_raw.shape[1]
            title_suffix = f" (All {n_particles} particles)"

        # Create 2x4 subplot grid
        fig = plt.figure(figsize=(16, 8))
        gs = fig.add_gridspec(2, 4, width_ratios=[1, 1, 1, 1], height_ratios=[1, 1])
        
        # Define subplot positions
        ax_x_linear = fig.add_subplot(gs[0, 0])      # (1,1): Linear x density
        ax_y_linear = fig.add_subplot(gs[0, 1])      # (1,2): Linear y density  
        ax_x_log = fig.add_subplot(gs[1, 0])         # (2,1): Log x density
        ax_y_log = fig.add_subplot(gs[1, 1])         # (2,2): Log y density
        ax_2d = fig.add_subplot(gs[:, 2:])           # (1-2,3-4): 2D histogram

        # Filter out NaN values before creating histograms
        valid_mask = ~np.any(np.isnan(positions), axis=1)
        valid_positions = positions[valid_mask]
        
        if len(valid_positions) == 0:
            raise ValueError("All position data contains NaN values - simulation failed")
        
        n_invalid = len(positions) - len(valid_positions)
        if n_invalid > 0:
            print(f"Warning: Filtered out {n_invalid} data points with NaN values")
        
        # Calculate histograms for 1D distributions
        x_counts, x_edges = np.histogram(valid_positions[:, 0], bins=50, density=True)
        y_counts, y_edges = np.histogram(valid_positions[:, 1], bins=50, density=True)
        
        # (1,1): Linear X density
        ax_x_linear.hist(
            valid_positions[:, 0],
            bins=50,
            density=True,
            alpha=0.7,
            color="blue",
            edgecolor="black",
        )
        ax_x_linear.set_xlabel("x position")
        ax_x_linear.set_ylabel("Density")
        ax_x_linear.set_title(f"X Position Density{title_suffix}")
        ax_x_linear.grid(True, alpha=0.3)

        # (1,2): Linear Y density
        ax_y_linear.hist(
            valid_positions[:, 1],
            bins=50,
            density=True,
            alpha=0.7,
            color="red",
            edgecolor="black",
        )
        ax_y_linear.set_xlabel("y position")
        ax_y_linear.set_ylabel("Density")
        ax_y_linear.set_title(f"Y Position Density{title_suffix}")
        ax_y_linear.grid(True, alpha=0.3)

        # (2,1): Log X density
        x_counts_nonzero = x_counts[x_counts > 0]  # Only plot where we have support
        x_edges_nonzero = x_edges[:-1][x_counts > 0]  # Corresponding bin edges
        x_counts_log = -np.log10(x_counts_nonzero)  # Negative log for energy-like representation
        
        ax_x_log.bar(
            x_edges_nonzero,
            x_counts_log,
            width=np.diff(x_edges)[0],  # Use consistent bin width
            alpha=0.7,
            color="blue",
            edgecolor="black",
        )
        ax_x_log.set_xlabel("x position")
        ax_x_log.set_ylabel("-Log₁₀(Density)")
        ax_x_log.set_title(f"X Position -Log Density{title_suffix}")
        ax_x_log.grid(True, alpha=0.3)

        # (2,2): Log Y density
        y_counts_nonzero = y_counts[y_counts > 0]  # Only plot where we have support
        y_edges_nonzero = y_edges[:-1][y_counts > 0]  # Corresponding bin edges
        y_counts_log = -np.log10(y_counts_nonzero)  # Negative log for energy-like representation
        
        ax_y_log.bar(
            y_edges_nonzero,
            y_counts_log,
            width=np.diff(y_edges)[0],  # Use consistent bin width
            alpha=0.7,
            color="red",
            edgecolor="black",
        )
        ax_y_log.set_xlabel("y position")
        ax_y_log.set_ylabel("-Log₁₀(Density)")
        ax_y_log.set_title(f"Y Position -Log Density{title_suffix}")
        ax_y_log.grid(True, alpha=0.3)

        # (1-2,3-4): 2D joint distribution (negative log scale)
        hist, xedges, yedges = np.histogram2d(
            valid_positions[:, 0], valid_positions[:, 1], bins=50, density=True
        )
        
        # Create negative log version, only where we have support
        hist_neglog = np.full_like(hist, np.nan)  # Initialize with NaN
        nonzero_mask = hist > 0
        hist_neglog[nonzero_mask] = -np.log10(hist[nonzero_mask])

        im = ax_2d.imshow(
            hist_neglog.T,
            origin="lower",
            extent=[xedges[0], xedges[-1], yedges[0], yedges[-1]],
            cmap="viridis_r",  # Reverse colormap so low values (high density) are bright
            alpha=0.8,
        )

        # Add critical points to 2D plot
        self._add_critical_points(ax_2d)

        ax_2d.set_xlabel("x position")
        ax_2d.set_ylabel("y position")
        ax_2d.set_title(f"2D Position Distribution{title_suffix}\n(Log Scale)")

        # Add colorbar for 2D plot
        cbar = fig.colorbar(im, ax=ax_2d, shrink=0.6)
        cbar.set_label("-Log(Density)", rotation=270, labelpad=15)

        plt.tight_layout()
        
        # Return axes in a format compatible with existing code
        axes = np.array([ax_x_linear, ax_y_linear, ax_x_log, ax_y_log, ax_2d])
        return fig, axes

    def plot_velocity_distributions(
        self, data: dict, sample_idx: int | None = None
    ) -> tuple[Figure, np.ndarray]:
        """Plot velocity distributions: components and magnitude.
        
        Args:
            data: Simulation data dictionary
            sample_idx: If provided, plot only for this particle index. If None (default),
                       aggregate across all particles.
                       
        Raises:
            ValueError: If velocities data is not available
        """
        if "velocities" not in data:
            raise ValueError("Velocity data not available - was 'velocities' included in saved observables?")
            
        velocities_raw = data["velocities"]  # (n_steps, n_particles, 2)
        
        if sample_idx is not None:
            # Single particle mode
            velocities = velocities_raw[:, sample_idx, :]  # (n_steps, 2)
            title_suffix = f" (Particle {sample_idx})"
        else:
            # All particles mode - flatten across particles and time steps
            velocities = velocities_raw.reshape(-1, 2)  # (n_steps * n_particles, 2)
            n_particles = velocities_raw.shape[1]
            title_suffix = f" (All {n_particles} particles)"
        
        # Filter out NaN values before creating histograms
        valid_mask = ~np.any(np.isnan(velocities), axis=1)
        valid_velocities = velocities[valid_mask]
        
        if len(valid_velocities) == 0:
            raise ValueError("All velocity data contains NaN values - simulation failed")
        
        n_invalid = len(velocities) - len(valid_velocities)
        if n_invalid > 0:
            print(f"Warning: Filtered out {n_invalid} velocity data points with NaN values")
            
        vel_magnitude = np.linalg.norm(valid_velocities, axis=1)

        fig, axes = plt.subplots(1, 3, figsize=DEFAULT_FIGURE_SIZE)

        # X velocity component
        axes[0].hist(
            valid_velocities[:, 0],
            bins=50,
            density=True,
            alpha=0.7,
            color="blue",
            edgecolor="black",
        )
        axes[0].set_xlabel("x velocity")
        axes[0].set_ylabel("Density")
        axes[0].set_title(f"X Velocity Distribution{title_suffix}")
        axes[0].grid(True, alpha=0.3)

        # Y velocity component
        axes[1].hist(
            valid_velocities[:, 1],
            bins=50,
            density=True,
            alpha=0.7,
            color="red",
            edgecolor="black",
        )
        axes[1].set_xlabel("y velocity")
        axes[1].set_ylabel("Density")
        axes[1].set_title(f"Y Velocity Distribution{title_suffix}")
        axes[1].grid(True, alpha=0.3)

        # Velocity magnitude
        axes[2].hist(
            vel_magnitude,
            bins=50,
            density=True,
            alpha=0.7,
            color="green",
            edgecolor="black",
        )
        axes[2].set_xlabel("velocity magnitude")
        axes[2].set_ylabel("Density")
        axes[2].set_title(f"Velocity Magnitude Distribution{title_suffix}")
        axes[2].grid(True, alpha=0.3)

        plt.tight_layout()
        return fig, axes

    def plot_force_distributions(
        self, data: dict, sample_idx: int | None = None
    ) -> tuple[Figure, np.ndarray]:
        """Plot force distributions: components and magnitude.
        
        Args:
            data: Simulation data dictionary
            sample_idx: If provided, plot only for this particle index. If None (default),
                       aggregate across all particles.
                       
        Raises:
            ValueError: If forces data is not available
        """
        if "forces" not in data:
            raise ValueError("Force data not available - was 'forces' included in saved observables?")
            
        forces_raw = data["forces"]  # (n_steps, n_particles, 2)
        
        if sample_idx is not None:
            # Single particle mode
            forces = forces_raw[:, sample_idx, :]  # (n_steps, 2)
            title_suffix = f" (Particle {sample_idx})"
        else:
            # All particles mode - flatten across particles and time steps
            forces = forces_raw.reshape(-1, 2)  # (n_steps * n_particles, 2)
            n_particles = forces_raw.shape[1]
            title_suffix = f" (All {n_particles} particles)"
        
        # Filter out NaN values before creating histograms
        valid_mask = ~np.any(np.isnan(forces), axis=1)
        valid_forces = forces[valid_mask]
        
        if len(valid_forces) == 0:
            raise ValueError("All force data contains NaN values - simulation failed")
        
        n_invalid = len(forces) - len(valid_forces)
        if n_invalid > 0:
            print(f"Warning: Filtered out {n_invalid} force data points with NaN values")
            
        force_magnitude = np.linalg.norm(valid_forces, axis=1)

        fig, axes = plt.subplots(1, 3, figsize=DEFAULT_FIGURE_SIZE)

        # X force component
        axes[0].hist(
            valid_forces[:, 0],
            bins=50,
            density=True,
            alpha=0.7,
            color="blue",
            edgecolor="black",
        )
        axes[0].set_xlabel("x force")
        axes[0].set_ylabel("Density")
        axes[0].set_title(f"X Force Distribution{title_suffix}")
        axes[0].grid(True, alpha=0.3)

        # Y force component
        axes[1].hist(
            valid_forces[:, 1],
            bins=50,
            density=True,
            alpha=0.7,
            color="red",
            edgecolor="black",
        )
        axes[1].set_xlabel("y force")
        axes[1].set_ylabel("Density")
        axes[1].set_title(f"Y Force Distribution{title_suffix}")
        axes[1].grid(True, alpha=0.3)

        # Force magnitude
        axes[2].hist(
            force_magnitude,
            bins=50,
            density=True,
            alpha=0.7,
            color="green",
            edgecolor="black",
        )
        axes[2].set_xlabel("force magnitude")
        axes[2].set_ylabel("Density")
        axes[2].set_title(f"Force Magnitude Distribution{title_suffix}")
        axes[2].grid(True, alpha=0.3)

        plt.tight_layout()
        return fig, axes

    def plot_energy_distribution(
        self, data: dict, sample_idx: int | None = None
    ) -> tuple[Figure, Axes]:
        """Plot potential energy distribution.
        
        Args:
            data: Simulation data dictionary
            sample_idx: If provided, plot only for this particle index. If None (default),
                       aggregate across all particles.
                       
        Raises:
            ValueError: If potential_energy data is not available
        """
        if "potential_energy" not in data:
            raise ValueError("Potential energy data not available - was 'potential_energy' included in saved observables?")
            
        energies_raw = data["potential_energy"]  # (n_steps, n_particles)
        
        if sample_idx is not None:
            # Single particle mode
            energies = energies_raw[:, sample_idx]  # (n_steps,)
            title_suffix = f" (Particle {sample_idx})"
        else:
            # All particles mode - flatten across particles and time steps
            energies = energies_raw.flatten()  # (n_steps * n_particles,)
            n_particles = energies_raw.shape[1]
            title_suffix = f" (All {n_particles} particles)"

        # Filter out NaN values before creating histogram
        valid_energies = energies[~np.isnan(energies)]
        
        if len(valid_energies) == 0:
            raise ValueError("All energy data contains NaN values - simulation failed")
        
        n_invalid = len(energies) - len(valid_energies)
        if n_invalid > 0:
            print(f"Warning: Filtered out {n_invalid} energy data points with NaN values")

        fig, ax = plt.subplots(figsize=(6, 4))

        ax.hist(
            valid_energies,
            bins=50,
            density=True,
            alpha=0.7,
            color="orange",
            edgecolor="black",
        )
        ax.set_xlabel("Potential Energy")
        ax.set_ylabel("Density")
        ax.set_title(f"Potential Energy Distribution{title_suffix}")
        ax.grid(True, alpha=0.3)

        return fig, ax

    def plot_batch_position_distributions(
        self, all_results: list[dict]
    ) -> tuple[Figure, np.ndarray]:
        """Plot position distributions aggregated across all particles and simulations.
        
        Args:
            all_results: List of simulation result dictionaries
            
        Raises:
            ValueError: If positions data is not available in any simulation
        """
        # Check if positions are available in all results
        missing_positions = [i for i, result in enumerate(all_results) if "positions" not in result]
        if missing_positions:
            raise ValueError(
                f"Position data not available in simulations {missing_positions} - "
                "was 'positions' included in saved observables?"
            )
        
        # Collect all positions from all simulations
        all_positions = []
        total_particles = 0
        total_simulations = len(all_results)
        
        for result in all_results:
            positions = result["positions"]  # (n_steps, n_particles, 2)
            all_positions.append(positions.reshape(-1, 2))  # Flatten to (n_steps * n_particles, 2)
            total_particles += positions.shape[1]
        
        # Concatenate all position data
        positions_combined = np.concatenate(all_positions, axis=0)  # (total_data_points, 2)
        
        # Filter out NaN values before creating histograms
        valid_mask = ~np.any(np.isnan(positions_combined), axis=1)
        valid_positions_combined = positions_combined[valid_mask]
        
        if len(valid_positions_combined) == 0:
            raise ValueError("All position data contains NaN values - simulations failed")
        
        n_invalid = len(positions_combined) - len(valid_positions_combined)
        if n_invalid > 0:
            print(f"Warning: Filtered out {n_invalid} position data points with NaN values from batch")
        
        avg_particles_per_sim = total_particles / total_simulations
        title_suffix = f" ({total_simulations} sims, avg {avg_particles_per_sim:.0f} particles each)"

        # Create 2x4 subplot grid
        fig = plt.figure(figsize=(16, 8))
        gs = fig.add_gridspec(2, 4, width_ratios=[1, 1, 1, 1], height_ratios=[1, 1])
        
        # Define subplot positions
        ax_x_linear = fig.add_subplot(gs[0, 0])      # (1,1): Linear x density
        ax_y_linear = fig.add_subplot(gs[0, 1])      # (1,2): Linear y density  
        ax_x_log = fig.add_subplot(gs[1, 0])         # (2,1): Log x density
        ax_y_log = fig.add_subplot(gs[1, 1])         # (2,2): Log y density
        ax_2d = fig.add_subplot(gs[:, 2:])           # (1-2,3-4): 2D histogram

        # Calculate histograms for 1D distributions
        x_counts, x_edges = np.histogram(valid_positions_combined[:, 0], bins=50, density=True)
        y_counts, y_edges = np.histogram(valid_positions_combined[:, 1], bins=50, density=True)
        
        # (1,1): Linear X density
        ax_x_linear.hist(
            valid_positions_combined[:, 0],
            bins=50,
            density=True,
            alpha=0.7,
            color="blue",
            edgecolor="black",
        )
        ax_x_linear.set_xlabel("x position")
        ax_x_linear.set_ylabel("Density")
        ax_x_linear.set_title(f"X Position Density{title_suffix}")
        ax_x_linear.grid(True, alpha=0.3)

        # (1,2): Linear Y density
        ax_y_linear.hist(
            valid_positions_combined[:, 1],
            bins=50,
            density=True,
            alpha=0.7,
            color="red",
            edgecolor="black",
        )
        ax_y_linear.set_xlabel("y position")
        ax_y_linear.set_ylabel("Density")
        ax_y_linear.set_title(f"Y Position Density{title_suffix}")
        ax_y_linear.grid(True, alpha=0.3)

        # (2,1): Log X density
        x_counts_nonzero = x_counts[x_counts > 0]  # Only plot where we have support
        x_edges_nonzero = x_edges[:-1][x_counts > 0]  # Corresponding bin edges
        x_counts_log = -np.log10(x_counts_nonzero)  # Negative log for energy-like representation
        
        ax_x_log.bar(
            x_edges_nonzero,
            x_counts_log,
            width=np.diff(x_edges)[0],  # Use consistent bin width
            alpha=0.7,
            color="blue",
            edgecolor="black",
        )
        ax_x_log.set_xlabel("x position")
        ax_x_log.set_ylabel("-Log₁₀(Density)")
        ax_x_log.set_title(f"X Position -Log Density{title_suffix}")
        ax_x_log.grid(True, alpha=0.3)

        # (2,2): Log Y density
        y_counts_nonzero = y_counts[y_counts > 0]  # Only plot where we have support
        y_edges_nonzero = y_edges[:-1][y_counts > 0]  # Corresponding bin edges
        y_counts_log = -np.log10(y_counts_nonzero)  # Negative log for energy-like representation
        
        ax_y_log.bar(
            y_edges_nonzero,
            y_counts_log,
            width=np.diff(y_edges)[0],  # Use consistent bin width
            alpha=0.7,
            color="red",
            edgecolor="black",
        )
        ax_y_log.set_xlabel("y position")
        ax_y_log.set_ylabel("-Log₁₀(Density)")
        ax_y_log.set_title(f"Y Position -Log Density{title_suffix}")
        ax_y_log.grid(True, alpha=0.3)

        # (1-2,3-4): 2D joint distribution (negative log scale)
        hist, xedges, yedges = np.histogram2d(
            valid_positions_combined[:, 0], valid_positions_combined[:, 1], bins=50, density=True
        )
        
        # Create negative log version, only where we have support
        hist_neglog = np.full_like(hist, np.nan)  # Initialize with NaN
        nonzero_mask = hist > 0
        hist_neglog[nonzero_mask] = -np.log10(hist[nonzero_mask])

        im = ax_2d.imshow(
            hist_neglog.T,
            origin="lower",
            extent=[xedges[0], xedges[-1], yedges[0], yedges[-1]],
            cmap="viridis_r",  # Reverse colormap so low values (high density) are bright
            alpha=0.8,
        )

        # Add critical points to 2D plot
        self._add_critical_points(ax_2d)

        ax_2d.set_xlabel("x position")
        ax_2d.set_ylabel("y position")
        ax_2d.set_title(f"2D Position Distribution{title_suffix}\n(-Log Scale)")

        # Add colorbar for 2D plot
        cbar = fig.colorbar(im, ax=ax_2d, shrink=0.6)
        cbar.set_label("-Log(Density)", rotation=270, labelpad=15)

        plt.tight_layout()
        
        # Return axes in a format compatible with existing code
        axes = np.array([ax_x_linear, ax_y_linear, ax_x_log, ax_y_log, ax_2d])
        return fig, axes

    def plot_trajectory_on_potential(
        self, data: dict, sample_idx: int = 0
    ) -> tuple[Figure, Axes]:
        """Plot single trajectory on potential surface.
        
        Raises:
            ValueError: If positions data is not available
        """
        if "positions" not in data:
            raise ValueError("Position data not available - was 'positions' included in saved observables?")
            
        positions = data["positions"][:, sample_idx, :]  # (n_steps, 2)

        # Create a single plot for trajectory on potential surface
        fig, ax = plt.subplots(figsize=(8, 6))

        # Plot trajectory on potential surface
        _, _ = self.plot_potential_surface(ax=ax)
        ax.plot(positions[:, 0], positions[:, 1], "white", linewidth=2.5, alpha=0.9, label="Trajectory")
        ax.plot(
            positions[0, 0], positions[0, 1], "ro", markersize=10, label="Start", zorder=12
        )
        ax.plot(
            positions[-1, 0], positions[-1, 1], "rs", markersize=10, label="End", zorder=12
        )
        
        # Improve legend styling
        ax.legend(
            loc="upper right",
            frameon=True,
            fancybox=True,
            shadow=True,
            framealpha=0.9,
            fontsize=10
        )
        ax.set_title(f"Trajectory on Potential Surface (Particle {sample_idx})", fontsize=14, fontweight='bold')

        plt.tight_layout()
        return fig, ax

    def plot_position_time_series(
        self, data: dict, sample_idx: int = 0
    ) -> tuple[Figure, np.ndarray]:
        """Plot x and y position time series separately.
        
        Raises:
            ValueError: If positions data is not available
        """
        if "positions" not in data:
            raise ValueError("Position data not available - was 'positions' included in saved observables?")
            
        positions = data["positions"][:, sample_idx, :]  # (n_steps, 2)
        dt = data["dt"]
        save_every = data.get("save_every", 1)
        n_transient_removed = data.get("n_transient_removed", 0)
        
        # Calculate time points accounting for transient removal
        time_offset = n_transient_removed * save_every * dt
        time_points = time_offset + np.arange(len(positions)) * save_every * dt

        fig, axes = plt.subplots(1, 2, figsize=(12, 4))

        # X position vs time
        axes[0].plot(time_points, positions[:, 0], "b-", linewidth=1.5)
        axes[0].set_xlabel("Time", fontsize=12)
        axes[0].set_ylabel("x position", fontsize=12)
        axes[0].set_title(f"X Position vs Time (Particle {sample_idx})", fontsize=13, fontweight='bold')
        axes[0].grid(True, alpha=0.3)
        axes[0].tick_params(labelsize=10)

        # Y position vs time
        axes[1].plot(time_points, positions[:, 1], "r-", linewidth=1.5)
        axes[1].set_xlabel("Time", fontsize=12)
        axes[1].set_ylabel("y position", fontsize=12)
        axes[1].set_title(f"Y Position vs Time (Particle {sample_idx})", fontsize=13, fontweight='bold')
        axes[1].grid(True, alpha=0.3)
        axes[1].tick_params(labelsize=10)

        plt.tight_layout()
        return fig, axes

    def create_animated_trajectory(
        self, 
        data: dict, 
        sample_idx: int = 0,
        output_path: str | Path | None = None,
        frames_per_second: int = 30,
        trail_length: int = 100,
        frame_skip: int = 1,
        show_energy: bool = True
    ) -> str:
        """Create an animated trajectory visualization as MP4 or GIF.
        
        Args:
            data: Simulation data dictionary
            sample_idx: Particle index to animate
            output_path: Path to save animation. If None, saves as 'trajectory_animation.mp4'
            frames_per_second: Animation frame rate
            trail_length: Number of recent positions to show as a trail
            frame_skip: Skip every N frames to reduce file size/animation length
            show_energy: Whether to show energy information in animation
            
        Returns:
            Path to saved animation file
            
        Raises:
            ValueError: If positions data is not available
            ImportError: If required animation dependencies are not available
        """
        try:
            from matplotlib.animation import FuncAnimation
        except ImportError:
            raise ImportError("matplotlib.animation is required for trajectory animation")
            
        if "positions" not in data:
            raise ValueError("Position data not available - was 'positions' included in saved observables?")
            
        positions = data["positions"][:, sample_idx, :]  # (n_steps, 2)
        dt = data["dt"]
        save_every = data.get("save_every", 1)
        n_transient_removed = data.get("n_transient_removed", 0)
        
        # Apply frame skipping to reduce animation length
        positions = positions[::frame_skip]
        
        # Calculate time points accounting for transient removal and frame skipping
        time_offset = n_transient_removed * save_every * dt
        time_points = time_offset + np.arange(len(positions)) * save_every * dt * frame_skip
        
        # Get potential energy if available and requested
        energies = None
        if show_energy and "potential_energy" in data:
            energies = data["potential_energy"][::frame_skip, sample_idx]
        
        # Set up the figure and axes for HD 720p (1280x720)
        # Calculate figsize in inches (matplotlib uses 100 DPI by default)
        fig_width = 1280 / 100  # 12.8 inches
        fig_height = 720 / 100  # 7.2 inches
        fig, ax = plt.subplots(figsize=(fig_width, fig_height), dpi=100)
        
        # Plot the potential surface
        self.plot_potential_surface(ax=ax)
        
        # Initialize trajectory line and current position marker
        trajectory_line, = ax.plot([], [], 'white', linewidth=2, alpha=0.7, label='Trajectory')
        current_pos, = ax.plot([], [], 'ro', markersize=10, label='Current Position')
        start_pos, = ax.plot([], [], 'go', markersize=8, label='Start')
        
        # Add title and legend
        title = ax.set_title(f'Trajectory Animation (Particle {sample_idx})')
        ax.legend(loc='upper right')
        
        # Text annotations for time and energy
        time_text = ax.text(0.02, 0.98, '', transform=ax.transAxes, 
                           verticalalignment='top', fontsize=12,
                           bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        energy_text = None
        if energies is not None:
            energy_text = ax.text(0.02, 0.90, '', transform=ax.transAxes,
                                 verticalalignment='top', fontsize=12,
                                 bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        def animate(frame):
            """Animation function for each frame."""
            # Current position
            current_pos.set_data([positions[frame, 0]], [positions[frame, 1]])
            
            # Show start position
            if frame == 0:
                start_pos.set_data([positions[0, 0]], [positions[0, 1]])
            
            # Show trail (recent trajectory)
            trail_start = max(0, frame - trail_length)
            trail_x = positions[trail_start:frame+1, 0]
            trail_y = positions[trail_start:frame+1, 1]
            trajectory_line.set_data(trail_x, trail_y)
            
            # Update time
            time_text.set_text(f'Time: {time_points[frame]:.3f}')
            
            # Update energy if available
            if energy_text is not None and energies is not None:
                energy_text.set_text(f'Energy: {energies[frame]:.3f}')
            
            # Update title with progress
            progress = (frame + 1) / len(positions) * 100
            title.set_text(f'Trajectory Animation (Particle {sample_idx}) - {progress:.1f}%')
            
            return trajectory_line, current_pos, start_pos, time_text, title
        
        # Create animation
        anim = FuncAnimation(
            fig, animate, frames=len(positions), 
            interval=1000/frames_per_second, blit=False, repeat=True
        )
        
        # Save animation
        if output_path is None:
            output_path = "trajectory_animation.mp4"
        
        output_path = Path(output_path)
        
        # Determine format from extension
        if output_path.suffix.lower() == '.gif':
            print(f"Saving animation as GIF: {output_path}")
            anim.save(output_path, writer='pillow', fps=frames_per_second)
        else:
            # Default to MP4 with high quality settings for HD
            if not output_path.suffix:
                output_path = output_path.with_suffix('.mp4')
            print(f"Saving animation as HD MP4 (1280x720): {output_path}")
            try:
                # High quality settings for HD video
                anim.save(output_path, writer='ffmpeg', fps=frames_per_second, 
                         bitrate=5000, extra_args=['-vcodec', 'libx264', '-pix_fmt', 'yuv420p'])
            except Exception as e:
                # Fallback to basic writer if ffmpeg is not available
                print(f"FFmpeg not available ({e}), trying alternative writer...")
                try:
                    anim.save(output_path, fps=frames_per_second, dpi=100)
                except Exception as e2:
                    print(f"Animation save failed: {e2}")
                    # Try saving as GIF instead
                    gif_path = output_path.with_suffix('.gif')
                    print(f"Falling back to GIF format: {gif_path}")
                    anim.save(gif_path, writer='pillow', fps=frames_per_second)
                    output_path = gif_path
        
        plt.close(fig)
        return str(output_path)

    def plot_velocity_time_series(
        self, data: dict, sample_idx: int = 0
    ) -> tuple[Figure, np.ndarray]:
        """Plot velocity components and magnitude vs time.
        
        Raises:
            ValueError: If velocities data is not available
        """
        if "velocities" not in data:
            raise ValueError("Velocity data not available - was 'velocities' included in saved observables?")
            
        velocities = data["velocities"][:, sample_idx, :]  # (n_steps, 2)
        dt = data["dt"]
        save_every = data.get("save_every", 1)
        n_transient_removed = data.get("n_transient_removed", 0)
        
        # Calculate time points accounting for transient removal
        time_offset = n_transient_removed * save_every * dt
        time_points = time_offset + np.arange(len(velocities)) * save_every * dt
        velocity_magnitude = np.linalg.norm(velocities, axis=1)

        fig, axes = plt.subplots(1, 3, figsize=(15, 4))

        # X velocity vs time
        axes[0].plot(time_points, velocities[:, 0], "b-", linewidth=1)
        axes[0].set_xlabel("Time")
        axes[0].set_ylabel("x velocity")
        axes[0].set_title("X Velocity vs Time")
        axes[0].grid(True, alpha=0.3)

        # Y velocity vs time
        axes[1].plot(time_points, velocities[:, 1], "r-", linewidth=1)
        axes[1].set_xlabel("Time")
        axes[1].set_ylabel("y velocity")
        axes[1].set_title("Y Velocity vs Time")
        axes[1].grid(True, alpha=0.3)

        # Velocity magnitude vs time
        axes[2].plot(time_points, velocity_magnitude, "g-", linewidth=1)
        axes[2].set_xlabel("Time")
        axes[2].set_ylabel("velocity magnitude")
        axes[2].set_title("Velocity Magnitude vs Time")
        axes[2].grid(True, alpha=0.3)

        plt.tight_layout()
        return fig, axes

    def plot_force_time_series(
        self, data: dict, sample_idx: int = 0
    ) -> tuple[Figure, np.ndarray]:
        """Plot force components and magnitude vs time.
        
        Raises:
            ValueError: If forces data is not available
        """
        if "forces" not in data:
            raise ValueError("Force data not available - was 'forces' included in saved observables?")
            
        forces = data["forces"][:, sample_idx, :]  # (n_steps, 2)
        dt = data["dt"]
        save_every = data.get("save_every", 1)
        n_transient_removed = data.get("n_transient_removed", 0)
        
        # Calculate time points accounting for transient removal
        time_offset = n_transient_removed * save_every * dt
        time_points = time_offset + np.arange(len(forces)) * save_every * dt
        force_magnitude = np.linalg.norm(forces, axis=1)

        fig, axes = plt.subplots(1, 3, figsize=(15, 4))

        # X force vs time
        axes[0].plot(time_points, forces[:, 0], "b-", linewidth=1)
        axes[0].set_xlabel("Time")
        axes[0].set_ylabel("x force")
        axes[0].set_title("X Force vs Time")
        axes[0].grid(True, alpha=0.3)

        # Y force vs time
        axes[1].plot(time_points, forces[:, 1], "r-", linewidth=1)
        axes[1].set_xlabel("Time")
        axes[1].set_ylabel("y force")
        axes[1].set_title("Y Force vs Time")
        axes[1].grid(True, alpha=0.3)

        # Force magnitude vs time
        axes[2].plot(time_points, force_magnitude, "g-", linewidth=1)
        axes[2].set_xlabel("Time")
        axes[2].set_ylabel("force magnitude")
        axes[2].set_title("Force Magnitude vs Time")
        axes[2].grid(True, alpha=0.3)

        plt.tight_layout()
        return fig, axes

    def plot_msd_vs_time(self, data: dict, sample_idx: int = 0) -> tuple[Figure, Axes]:
        """Plot mean squared displacement vs time.
        
        Raises:
            ValueError: If positions data is not available
        """
        if "positions" not in data:
            raise ValueError("Position data not available - was 'positions' included in saved observables?")
            
        positions = data["positions"][:, sample_idx, :]  # (n_steps, 2)
        dt = data["dt"]
        save_every = data.get("save_every", 1)
        n_transient_removed = data.get("n_transient_removed", 0)
        
        # Calculate time points accounting for transient removal
        time_offset = n_transient_removed * save_every * dt
        time_points = time_offset + np.arange(len(positions)) * save_every * dt

        # Calculate MSD from initial position
        initial_pos = positions[0]
        msd = np.sum((positions - initial_pos) ** 2, axis=1)

        fig, ax = plt.subplots(figsize=(8, 4))

        ax.plot(time_points, msd, "g-", linewidth=2)
        ax.set_xlabel("Time")
        ax.set_ylabel("Mean Squared Displacement")
        ax.set_title("MSD vs Time")
        ax.grid(True, alpha=0.3)

        return fig, ax

    def plot_energy_vs_time(
        self, data: dict, sample_idx: int = 0
    ) -> tuple[Figure, Axes]:
        """Plot potential energy vs time.
        
        Raises:
            ValueError: If potential_energy data is not available
        """
        if "potential_energy" not in data:
            raise ValueError("Potential energy data not available - was 'potential_energy' included in saved observables?")
            
        energies = data["potential_energy"][:, sample_idx]  # (n_steps,)
        dt = data["dt"]
        save_every = data.get("save_every", 1)
        n_transient_removed = data.get("n_transient_removed", 0)
        
        # Calculate time points accounting for transient removal
        time_offset = n_transient_removed * save_every * dt
        time_points = time_offset + np.arange(len(energies)) * save_every * dt

        fig, ax = plt.subplots(figsize=(8, 4))

        ax.plot(time_points, energies, "orange", linewidth=1)
        ax.set_xlabel("Time")
        ax.set_ylabel("Potential Energy")
        ax.set_title("Potential Energy vs Time")
        ax.grid(True, alpha=0.3)

        return fig, ax

    # Helper methods
    def _compute_potential_grid(self, X: np.ndarray, Y: np.ndarray) -> np.ndarray:
        """Compute potential energy on a coordinate grid using batched processing."""
        coordinates = convert_to_tensor(
            np.stack([X.ravel(), Y.ravel()], axis=1), dtype=torch.float64
        )

        V_list = []
        with torch.no_grad():
            for i in range(0, len(coordinates), DEFAULT_BATCH_SIZE):
                batch_coords = coordinates[i : i + DEFAULT_BATCH_SIZE]
                V_batch = self.potential(batch_coords).numpy()
                V_list.append(V_batch)

        return np.concatenate(V_list).reshape(X.shape)

    def _apply_log_transform(self, V: np.ndarray) -> np.ndarray:
        """Apply log transformation for better energy surface visualization."""
        V_min = V.min()
        V_log = np.log(V - V_min + 1e-6)
        return np.clip(V_log, 3, 7)  # Clamp for better contrast

    def _setup_figure_and_axes(
        self, ax: Axes | None, figsize: tuple[float, float]
    ) -> tuple[Figure, Axes]:
        """Set up figure and axes with consistent formatting."""
        if ax is None:
            fig, ax = plt.subplots(figsize=figsize)
        else:
            fig = ax.figure
        return fig, ax

    def _add_critical_points(self, ax: Axes) -> None:
        """Add minima and saddle points to the plot with clear annotations."""
        # Plot minima with consistent styling
        minima = self.potential.get_minima()
        for i, (x, y) in enumerate(minima):
            ax.plot(
                x,
                y,
                "o",
                color="black",
                markersize=10,
                markerfacecolor="white",
                markeredgewidth=2.5,
                label="Minima" if i == 0 else "",
                zorder=10,
            )
            # Add text labels with better positioning and styling
            ax.annotate(
                f"M{chr(65 + i)}",
                (x, y),
                xytext=(8, 8),
                textcoords="offset points",
                fontweight="bold",
                fontsize=11,
                color="black",
                bbox=dict(
                    boxstyle="round,pad=0.3",
                    facecolor="white",
                    edgecolor="black",
                    alpha=0.9
                ),
                zorder=11,
            )

        # Plot saddle points with distinct styling
        saddles = self.potential.get_saddle_points()
        for i, (x, y) in enumerate(saddles):
            ax.plot(
                x,
                y,
                "X",
                color="black",
                markersize=12,
                markerfacecolor="white",
                markeredgewidth=2.5,
                label="Saddle Points" if i == 0 else "",
                zorder=10,
            )
            # Add text labels with better positioning and styling
            ax.annotate(
                f"S{i + 1}",
                (x, y),
                xytext=(8, 8),
                textcoords="offset points",
                fontweight="bold",
                fontsize=11,
                color="black",
                bbox=dict(
                    boxstyle="round,pad=0.3",
                    facecolor="white",
                    edgecolor="black",
                    alpha=0.9
                ),
                zorder=11,
            )

        # Add a clean legend for critical points
        handles, labels = ax.get_legend_handles_labels()
        critical_handles = []
        critical_labels = []
        
        # Find the handles for minima and saddle points
        for handle, label in zip(handles, labels):
            if label in ["Minima", "Saddle Points"]:
                critical_handles.append(handle)
                critical_labels.append(label)
        
        if critical_handles:
            ax.legend(
                critical_handles, 
                critical_labels, 
                loc="upper right",
                frameon=True,
                fancybox=True,
                shadow=True,
                framealpha=0.9,
                fontsize=10
            )
