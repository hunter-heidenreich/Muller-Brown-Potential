#!/usr/bin/env python3
"""Performance benchmark comparing analytical vs autograd force calculations."""

import time
import numpy as np
import torch
import matplotlib.pyplot as plt
from pathlib import Path
import json
from dataclasses import dataclass, asdict
from typing import List
from src.muller_brown import MuellerBrownPotential


@dataclass
class BenchmarkResult:
    """Container for benchmark results."""

    method: str
    n_particles: int
    n_iterations: int
    wall_time: float
    time_per_iteration: float
    time_per_particle_per_iteration: float
    throughput_particles_per_second: float


class PerformanceBenchmark:
    """Performance benchmark suite for force calculations."""

    def __init__(self, device: str = "cpu", dtype: torch.dtype = torch.float64):
        self.device = device
        self.dtype = dtype
        self.results: List[BenchmarkResult] = []

        # Initialize potentials
        self.potential_analytical = MuellerBrownPotential(
            device=device, dtype=dtype, use_autograd=False
        )
        self.potential_autograd = MuellerBrownPotential(
            device=device, dtype=dtype, use_autograd=True
        )

    def _benchmark_method(
        self,
        potential: MuellerBrownPotential,
        coordinates: torch.Tensor,
        method_name: str,
        n_iterations: int = 1000,
        warmup: int = 100,
        n_runs: int = 5,
    ) -> BenchmarkResult:
        """Benchmark a single method with multiple runs for statistical accuracy."""

        n_particles = coordinates.shape[0]

        # Warmup
        for _ in range(warmup):
            _ = potential.force(coordinates)

        # Multiple timing runs
        times = []
        for _ in range(n_runs):
            if self.device == "cuda":
                torch.cuda.synchronize()

            start = time.perf_counter()
            for _ in range(n_iterations):
                _ = potential.force(coordinates)

            if self.device == "cuda":
                torch.cuda.synchronize()

            times.append(time.perf_counter() - start)

        # Use median time for robustness
        wall_time = np.median(times)
        time_per_iteration = wall_time / n_iterations
        time_per_particle_per_iteration = time_per_iteration / n_particles
        throughput = n_particles * n_iterations / wall_time

        return BenchmarkResult(
            method=method_name,
            n_particles=n_particles,
            n_iterations=n_iterations,
            wall_time=wall_time,
            time_per_iteration=time_per_iteration,
            time_per_particle_per_iteration=time_per_particle_per_iteration,
            throughput_particles_per_second=throughput,
        )

    def run_benchmarks(self, max_particles: int = 50000) -> None:
        """Run performance benchmarks across different problem sizes."""
        print("🔧 Running Performance Benchmarks")
        print("=" * 50)

        # Particle counts (powers of 2 and 10)
        particle_counts = []
        for base in [2, 10]:
            power = 1
            while base**power <= max_particles:
                particle_counts.append(base**power)
                power += 1

        particle_counts = sorted(set(particle_counts))
        print(f"Testing particle counts: {particle_counts}")
        print()

        for n_particles in particle_counts:
            print(f"Testing {n_particles:,} particles...")

            # Generate test coordinates
            coordinates = self._generate_coordinates(n_particles)

            # Adjust iterations based on problem size
            n_iterations = max(100, 50000 // n_particles)

            # Benchmark both methods
            result_analytical = self._benchmark_method(
                self.potential_analytical, coordinates, "analytical", n_iterations
            )
            result_autograd = self._benchmark_method(
                self.potential_autograd, coordinates, "autograd", n_iterations
            )

            self.results.extend([result_analytical, result_autograd])

            # Print progress
            speedup = result_autograd.wall_time / result_analytical.wall_time
            print(
                f"  Analytical: {result_analytical.throughput_particles_per_second:8.0f} particles/s"
            )
            print(
                f"  Autograd:   {result_autograd.throughput_particles_per_second:8.0f} particles/s"
            )
            print(f"  Speedup:    {speedup:8.1f}x")
            print()

    def _generate_coordinates(self, n_particles: int) -> torch.Tensor:
        """Generate random coordinates in the Müller-Brown relevant region."""
        torch.manual_seed(42)  # Reproducible results
        coordinates = (
            torch.randn(n_particles, 2, device=self.device, dtype=self.dtype) * 0.5
        )
        coordinates[:, 0] += -0.2  # Center around interesting region
        coordinates[:, 1] += 0.5
        return coordinates

    def create_plots(self, output_dir: str = "artifacts/benchmark_plots") -> None:
        """Create performance analysis plots."""
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)

        # Separate analytical and autograd results
        analytical = [r for r in self.results if r.method == "analytical"]
        autograd = [r for r in self.results if r.method == "autograd"]

        self._plot_throughput_analysis(
            analytical, autograd, output_path / "throughput_analysis.png"
        )
        self._plot_time_per_particle(
            analytical, autograd, output_path / "time_per_particle.png"
        )

        print(f"📊 Plots saved to {output_path}")

    def _plot_throughput_analysis(
        self, analytical: List, autograd: List, output_path: Path
    ):
        """Plot throughput vs problem size."""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

        # Extract data
        analytical_particles = [r.n_particles for r in analytical]
        analytical_throughput = [r.throughput_particles_per_second for r in analytical]
        autograd_particles = [r.n_particles for r in autograd]
        autograd_throughput = [r.throughput_particles_per_second for r in autograd]

        # Left plot: Log-log scale
        ax1.loglog(
            analytical_particles,
            analytical_throughput,
            "o-",
            label="Analytical",
            color="#FF6B6B",
            linewidth=2,
            markersize=6,
        )
        ax1.loglog(
            autograd_particles,
            autograd_throughput,
            "s-",
            label="Autograd",
            color="#4ECDC4",
            linewidth=2,
            markersize=6,
        )
        ax1.set_xlabel("Number of Particles")
        ax1.set_ylabel("Throughput (particles/second)")
        ax1.set_title("Throughput vs Problem Size (Log-Log)")
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # Right plot: Semi-log scale
        ax2.semilogx(
            analytical_particles,
            np.array(analytical_throughput) / 1e7,
            "o-",
            label="Analytical",
            color="#FF6B6B",
            linewidth=2,
            markersize=6,
        )
        ax2.semilogx(
            autograd_particles,
            np.array(autograd_throughput) / 1e7,
            "s-",
            label="Autograd",
            color="#4ECDC4",
            linewidth=2,
            markersize=6,
        )
        ax2.set_xlabel("Number of Particles")
        ax2.set_ylabel("Throughput (×10⁷ particles/second)")
        ax2.set_title("Throughput vs Problem Size (Semi-Log)")
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        plt.close()

    def _plot_time_per_particle(
        self, analytical: List, autograd: List, output_path: Path
    ):
        """Plot time per particle vs problem size."""
        fig, ax = plt.subplots(1, 1, figsize=(10, 6))

        # Extract data
        analytical_particles = [r.n_particles for r in analytical]
        analytical_time = [
            r.time_per_particle_per_iteration * 1e6 for r in analytical
        ]  # Convert to microseconds
        autograd_particles = [r.n_particles for r in autograd]
        autograd_time = [r.time_per_particle_per_iteration * 1e6 for r in autograd]

        ax.loglog(
            analytical_particles,
            analytical_time,
            "o-",
            label="Analytical",
            color="#FF6B6B",
            linewidth=2,
            markersize=6,
        )
        ax.loglog(
            autograd_particles,
            autograd_time,
            "s-",
            label="Autograd",
            color="#4ECDC4",
            linewidth=2,
            markersize=6,
        )
        ax.set_xlabel("Number of Particles")
        ax.set_ylabel("Time per Particle per Iteration (μs)")
        ax.set_title("Per-Particle Performance vs Problem Size")
        ax.legend()
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        plt.close()

    def save_results(self, filename: str = "artifacts/benchmark_results.json"):
        """Save benchmark results to JSON."""
        data = [asdict(result) for result in self.results]
        with open(filename, "w") as f:
            json.dump(data, f, indent=2)
        print(f"💾 Results saved to {filename}")

    def print_summary(self):
        """Print performance summary."""
        print("📈 Performance Summary")
        print("=" * 50)

        # Group by problem size
        analytical = [r for r in self.results if r.method == "analytical"]
        autograd = [r for r in self.results if r.method == "autograd"]

        print(f"{'Particles':<10} {'Analytical':<12} {'Autograd':<12} {'Speedup':<8}")
        print("-" * 50)

        for a_result, g_result in zip(analytical, autograd):
            speedup = g_result.wall_time / a_result.wall_time
            print(
                f"{a_result.n_particles:<10,} {a_result.throughput_particles_per_second:<12,.0f} "
                f"{g_result.throughput_particles_per_second:<12,.0f} {speedup:<8.1f}x"
            )

        if analytical:
            avg_speedup = np.mean(
                [g.wall_time / a.wall_time for a, g in zip(analytical, autograd)]
            )
            max_throughput_analytical = max(
                r.throughput_particles_per_second for r in analytical
            )
            max_throughput_autograd = max(
                r.throughput_particles_per_second for r in autograd
            )

            print("-" * 50)
            print(f"Average speedup: {avg_speedup:.1f}x")
            print(
                f"Peak analytical throughput: {max_throughput_analytical:,.0f} particles/s"
            )
            print(
                f"Peak autograd throughput: {max_throughput_autograd:,.0f} particles/s"
            )


def main():
    """Run performance benchmark suite."""
    print("🚀 Müller-Brown Force Calculation Performance Benchmark")
    print("=" * 60)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    print(f"PyTorch version: {torch.__version__}")
    if device == "cuda":
        print(f"CUDA device: {torch.cuda.get_device_name()}")
    print()

    # Initialize benchmark suite
    benchmark = PerformanceBenchmark(device=device, dtype=torch.float64)

    # Run benchmarks
    benchmark.run_benchmarks(max_particles=50000)

    # Create plots
    benchmark.create_plots()

    # Save results and print summary
    benchmark.save_results()
    benchmark.print_summary()

    print("\n📊 Visualization plots created in 'benchmark_plots/' directory")


if __name__ == "__main__":
    main()
