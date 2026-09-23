"""Make the project's src-layout package importable for these tests."""
import sys
from pathlib import Path

SRC = str(Path(__file__).resolve().parent.parent / 'src')
if SRC not in sys.path:
    sys.path.insert(0, SRC)
