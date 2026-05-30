"""Shared test configuration: force a non-interactive matplotlib backend so the
visualization tests render without a display."""

import matplotlib

matplotlib.use("Agg")
