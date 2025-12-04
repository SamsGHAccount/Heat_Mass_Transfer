import sys
from pathlib import Path

# Ensure the src/ directory (which contains conduction.py, fins.py, etc.)
# is on sys.path so tests can do `import conduction`, `import fins`, ...
ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))