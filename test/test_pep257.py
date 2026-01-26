"""ament_pep257 linter test."""

import pytest


try:
    from ament_pep257.main import main

    HAVE_AMENT_PEP257 = True
except Exception:  # pragma: no cover
    HAVE_AMENT_PEP257 = False


@pytest.mark.linter
@pytest.mark.pep257
def test_pep257():
    if not HAVE_AMENT_PEP257:
        pytest.skip("ament_pep257 not available in this environment")
    rc = main(argv=["."])
    assert rc == 0, "Found code style errors / warnings"
