"""Modern visualization tools for the Müller-Brown potential and simulation data."""

import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.axes import Axes
from matplotlib.figure import Figure

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
        fig, ax = self._setup_figure_and_axes(ax, (8, 6))

        # Plot contours
        v_min, v_max = V_log.min(), V_log.max()
        contour_levels = np.linspace(v_min, v_max, levels)

        contour = ax.contourf(
            X, Y, V_log, levels=contour_levels, cmap=cmap, extend="both"
        )

        # Add colorbar
        cbar = fig.colorbar(contour, ax=ax)
        cbar.set_label(r"$\log(V - V_{\min})$", rotation=0, labelpad=20)

        # Add critical points
        self._add_critical_points(ax)

        # Labels and title
        ax.set_xlabel(r"$x$", fontsize=12)
        ax.set_ylabel(r"$y$", fontsize=12)
        ax.set_title("Müller-Brown Potential Energy Surface", fontsize=14)
        ax.grid(True, alpha=0.3)

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
    ) -> tuple[Figure, np.ndarray]:
        """Plot single trajectory on potential surface with time series for x and y.
        
        Raises:
            ValueError: If positions data is not available
        """
        if "positions" not in data:
            raise ValueError("Position data not available - was 'positions' included in saved observables?")
            
        positions = data["positions"][:, sample_idx, :]  # (n_steps, 2)
        dt = data["dt"]
        time_points = np.arange(len(positions)) * dt

        fig, axes = plt.subplots(1, 3, figsize=(15, 4))

        # Plot trajectory on potential surface
        _, _ = self.plot_potential_surface(ax=axes[0])
        axes[0].plot(positions[:, 0], positions[:, 1], "white", linewidth=2, alpha=0.8)
        axes[0].plot(
            positions[0, 0], positions[0, 1], "ro", markersize=8, label="Start"
        )
        axes[0].plot(
            positions[-1, 0], positions[-1, 1], "rs", markersize=8, label="End"
        )
        axes[0].legend()
        axes[0].set_title("Trajectory on Potential Surface")

        # X position vs time
        axes[1].plot(time_points, positions[:, 0], "b-", linewidth=1)
        axes[1].set_xlabel("Time")
        axes[1].set_ylabel("x position")
        axes[1].set_title("X Position vs Time")
        axes[1].grid(True, alpha=0.3)

        # Y position vs time
        axes[2].plot(time_points, positions[:, 1], "r-", linewidth=1)
        axes[2].set_xlabel("Time")
        axes[2].set_ylabel("y position")
        axes[2].set_title("Y Position vs Time")
        axes[2].grid(True, alpha=0.3)

        plt.tight_layout()
        return fig, axes

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
        time_points = np.arange(len(velocities)) * dt
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
        time_points = np.arange(len(forces)) * dt
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
        time_points = np.arange(len(positions)) * dt

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
        time_points = np.arange(len(energies)) * dt

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
        """Add minima and saddle points to the plot."""
        # Plot minima
        minima = self.potential.get_minima()
        for i, (x, y) in enumerate(minima):
            ax.plot(
                x,
                y,
                "ko",
                markersize=8,
                markerfacecolor="white",
                markeredgewidth=2,
                label=f"Minimum {chr(65 + i)}" if i == 0 else None,
            )
            ax.annotate(
                f"M{chr(65 + i)}",
                (x, y),
                xytext=(5, 5),
                textcoords="offset points",
                fontweight="bold",
            )

        # Plot saddle points
        saddles = self.potential.get_saddle_points()
        for i, (x, y) in enumerate(saddles):
            ax.plot(
                x,
                y,
                "kX",
                markersize=10,
                markerfacecolor="white",
                markeredgewidth=2,
                label="Saddle Point" if i == 0 else None,
            )
            ax.annotate(
                f"S{i + 1}",
                (x, y),
                xytext=(5, 5),
                textcoords="offset points",
                fontweight="bold",
            )

        # Add legend for critical points only
        handles, labels = ax.get_legend_handles_labels()
        if handles:
            ax.legend(handles[:2], labels[:2], loc="upper right")
