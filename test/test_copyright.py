"""ament_copyright linter test.

The default ament_python template skips this check unless project-specific
copyright headers are added.
"""

import pytest


@pytest.mark.copyright
@pytest.mark.linter
def test_copyright():
	pytest.skip("No copyright header has been placed in the generated source files.")

