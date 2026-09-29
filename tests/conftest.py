"""Test paths and markers for read-only real data."""

def pytest_configure(config):
    config.addinivalue_line("markers", "real_data: Requires explicitly specified external read-only calculation data")
