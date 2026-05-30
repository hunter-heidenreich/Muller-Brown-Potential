"""The public API stays consistent: everything in __all__ is importable."""

import src.muller_brown as mb


def test_all_exports_are_importable():
    missing = [name for name in mb.__all__ if not hasattr(mb, name)]
    assert not missing, f"names listed in __all__ but not importable: {missing}"
