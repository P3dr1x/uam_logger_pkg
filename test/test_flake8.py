"""ament_flake8 linter test."""

import pytest


try:
    from ament_flake8.main import main_with_errors

    HAVE_AMENT_FLAKE8 = True
except Exception:  # pragma: no cover
    HAVE_AMENT_FLAKE8 = False


@pytest.mark.flake8
@pytest.mark.linter
def test_flake8():
    if not HAVE_AMENT_FLAKE8:
        pytest.skip("ament_flake8 not available in this environment")
    rc, errors = main_with_errors(argv=[])
    assert rc == 0, "Found %d code style errors / warnings:\n" % len(errors) + "\n".join(errors)
