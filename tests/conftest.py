"""Test paths and markers for read-only real data."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

def pytest_configure(config):
    config.addinivalue_line("markers", "real_data: Requires explicitly specified external read-only calculation data")
