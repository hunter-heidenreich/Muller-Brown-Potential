"""Visualization tools for the Müller-Brown potential and trajectories."""

from typing import Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.axes import Axes
from matplotlib.figure import Figure

from .potential import MuellerBrownPotential


class PotentialVisualizer:
    """Visualization tools for Müller-Brown potential energy surface and trajectories."""
    
    def __init__(self, potential: MuellerBrownPotential):
        """
        Initialize the visualizer.
        
        Args:
            potential: Müller-Brown potential instance
        """
        self.potential = potential
        
    def plot_potential(
        self,
        x_range: Tuple[float, float] = (-2.0, 2.0),
        y_range: Tuple[float, float] = (-1.0, 3.0),
        resolution: int = 200,  # Reduced default resolution
        ax: Optional[Axes] = None,
        cmap: str = "Reds",
        levels: int = 24,
    ) -> Tuple[Figure, Axes]:
        """
        Plot the Müller-Brown potential energy surface.
        Optimized version with reduced memory usage.
        
        Args:
            x_range: Range for x-axis
            y_range: Range for y-axis
            resolution: Grid resolution
            ax: Matplotlib axes (optional)
            cmap: Colormap name
            levels: Number of contour levels
            
        Returns:
            Figure and axes objects
        """
        # Create coordinate grid with optimized spacing
        x = np.linspace(x_range[0], x_range[1], resolution)
        y = np.linspace(y_range[0], y_range[1], resolution)
        X, Y = np.meshgrid(x, y)
        
        # Compute potential on grid in batches to save memory
        coordinates = torch.tensor(np.stack([X.ravel(), Y.ravel()], axis=1), dtype=torch.float64)
        
        batch_size = 10000  # Process in batches
        V_list = []
        
        with torch.no_grad():
            for i in range(0, len(coordinates), batch_size):
                batch_coords = coordinates[i:i+batch_size]
                V_batch = self.potential(batch_coords).numpy()
                V_list.append(V_batch)
        
        V = np.concatenate(V_list).reshape(X.shape)
        
        # Apply log transformation for better visualization
        V_min = V.min()
        V_log = np.log(V - V_min + 1e-6)
        V_log = np.clip(V_log, 3, 7)  # Clamp for better contrast
        
        # Create plot
        if ax is None:
            fig, ax = plt.subplots(figsize=(10, 8))
        else:
            fig = ax.figure
            
        # Plot contours
        v_min, v_max = V_log.min(), V_log.max()
        contour_levels = np.linspace(v_min, v_max, levels)
        
        contour = ax.contourf(X, Y, V_log, levels=contour_levels, cmap=cmap, extend='both')
        
        # Add colorbar
        cbar = fig.colorbar(contour, ax=ax)
        cbar.set_label(r'$\log(V - V_{\min})$', rotation=0, labelpad=20)
        
        # Add critical points
        self._add_critical_points(ax)
        
        # Labels and title
        ax.set_xlabel(r'$x_1$', fontsize=12)
        ax.set_ylabel(r'$x_2$', fontsize=12)
        ax.set_title('Müller-Brown Potential Energy Surface', fontsize=14)
        ax.grid(True, alpha=0.3)
        
        return fig, ax
    
    def _add_critical_points(self, ax: Axes) -> None:
        """Add minima and saddle points to the plot."""
        # Plot minima
        minima = self.potential.get_minima()
        for i, (x, y) in enumerate(minima):
            ax.plot(x, y, 'ko', markersize=8, markerfacecolor='white', 
                   markeredgewidth=2, label=f'Minimum {chr(65+i)}' if i == 0 else None)
            ax.annotate(f'M{chr(65+i)}', (x, y), xytext=(5, 5), 
                       textcoords='offset points', fontweight='bold')
        
        # Plot saddle points
        saddles = self.potential.get_saddle_points()
        for i, (x, y) in enumerate(saddles):
            ax.plot(x, y, 'kX', markersize=10, markerfacecolor='white', 
                   markeredgewidth=2, label='Saddle Point' if i == 0 else None)
            ax.annotate(f'S{i+1}', (x, y), xytext=(5, 5), 
                       textcoords='offset points', fontweight='bold')
        
        # Add legend for critical points only
        handles, labels = ax.get_legend_handles_labels()
        if handles:
            ax.legend(handles[:2], labels[:2], loc='upper right')
    
    def plot_trajectories(
        self,
        trajectories: np.ndarray,
        ax: Optional[Axes] = None,
        plot_potential: bool = True,
        trajectory_kwargs: Optional[dict] = None,
        **potential_kwargs,
    ) -> Tuple[Figure, Axes]:
        """
        Plot trajectories on the potential energy surface.
        
        Args:
            trajectories: Array of shape (n_particles, n_steps, 2)
            ax: Matplotlib axes (optional)
            plot_potential: Whether to plot the potential surface
            trajectory_kwargs: Keyword arguments for trajectory plotting
            **potential_kwargs: Keyword arguments for potential plotting
            
        Returns:
            Figure and axes objects
        """
        if trajectory_kwargs is None:
            trajectory_kwargs = {}
        
        # Default trajectory plotting parameters
        traj_defaults = {
            'alpha': 0.7,
            'linewidth': 1.5,
            'marker': 'o',
            'markersize': 2,
            'markevery': max(1, trajectories.shape[1] // 20),  # Show ~20 markers per trajectory
        }
        traj_defaults.update(trajectory_kwargs)
        
        # Use default potential plot bounds for consistency
        default_x_range = (-2.0, 2.0)
        default_y_range = (-1.0, 3.0)
        
        # Plot potential surface if requested
        if plot_potential:
            potential_defaults = {
                'x_range': potential_kwargs.get('x_range', default_x_range),
                'y_range': potential_kwargs.get('y_range', default_y_range)
            }
            potential_defaults.update(potential_kwargs)
            fig, ax = self.plot_potential(ax=ax, **potential_defaults)
        else:
            if ax is None:
                fig, ax = plt.subplots(figsize=(10, 8))
            else:
                fig = ax.figure
            
            # Set the same bounds as the default potential plot
            x_range = potential_kwargs.get('x_range', default_x_range)
            y_range = potential_kwargs.get('y_range', default_y_range)
            ax.set_xlim(x_range)
            ax.set_ylim(y_range)
        
        # Plot trajectories
        for i, trajectory in enumerate(trajectories):
            color = plt.cm.tab10(i % 10)  # Cycle through colors
            ax.plot(trajectory[:, 0], trajectory[:, 1], color=color, **traj_defaults)
            
            # Mark start and end points
            ax.plot(trajectory[0, 0], trajectory[0, 1], 'o', color=color, 
                   markersize=6, markerfacecolor='white', markeredgewidth=2, 
                   label='Start' if i == 0 else None)
            ax.plot(trajectory[-1, 0], trajectory[-1, 1], 's', color=color, 
                   markersize=6, markerfacecolor=color, markeredgewidth=1,
                   label='End' if i == 0 else None)
        
        # Add legend for trajectory markers
        if trajectories.shape[0] > 0:
            handles, labels = ax.get_legend_handles_labels()
            # Keep only start/end markers in legend, not all critical points
            start_end_handles = [h for h, label in zip(handles, labels) if label in ['Start', 'End']]
            start_end_labels = [label for label in labels if label in ['Start', 'End']]
            if start_end_handles:
                ax.legend(start_end_handles, start_end_labels, loc='upper left')
        
        ax.set_title(f'Trajectories on Müller-Brown Potential ({len(trajectories)} particles)')
        
        return fig, ax
    
    def plot_energy_vs_time(
        self,
        trajectory_data: dict,
        ax: Optional[Axes] = None,
    ) -> Tuple[Figure, Axes]:
        """
        Plot potential energy evolution during simulation.
        Optimized version with batched energy computation.
        
        Args:
            trajectory_data: Dictionary from LangevinSimulator.simulate()
            ax: Matplotlib axes (optional)
            
        Returns:
            Figure and axes objects
        """
        if ax is None:
            fig, ax = plt.subplots(figsize=(10, 6))
        else:
            fig = ax.figure
            
        trajectories = trajectory_data['trajectory']  # Shape: (n_steps, n_particles, 2)
        dt = trajectory_data['dt']
        
        # Compute energies for each time step efficiently
        n_steps, n_particles = trajectories.shape[:2]
        time_points = np.arange(n_steps) * dt
        
        # Batch compute all energies at once
        all_coords = torch.tensor(trajectories.reshape(-1, 2), dtype=torch.float64)
        
        with torch.no_grad():
            all_energies = self.potential(all_coords).numpy()
        
        energies = all_energies.reshape(n_steps, n_particles)
        
        # Plot energy traces
        for i in range(n_particles):
            color = plt.cm.tab10(i % 10)
            ax.plot(time_points, energies[:, i], color=color, alpha=0.7, linewidth=1)
        
        # Plot average energy
        mean_energy = energies.mean(axis=1)
        ax.plot(time_points, mean_energy, 'k-', linewidth=2, label='Average')
        
        ax.set_xlabel('Time')
        ax.set_ylabel('Potential Energy')
        ax.set_title('Energy Evolution During Simulation')
        ax.grid(True, alpha=0.3)
        ax.legend()
        
        return fig, ax
