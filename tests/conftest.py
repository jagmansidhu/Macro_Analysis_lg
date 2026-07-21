import sys
from pathlib import Path

from dotenv import load_dotenv

# Load .env before anything else so DB_CONNECTION_STRING and API keys are available
load_dotenv(Path(__file__).resolve().parent.parent / ".env")

import pytest  # noqa: E402

# Make src/ importable from any test
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

PROJECT_ROOT = Path(__file__).resolve().parent.parent


@pytest.fixture(scope="session")
def anyio_backend() -> str:
    return "asyncio"


@pytest.fixture(scope="session")
def project_root() -> Path:
    return PROJECT_ROOT


@pytest.fixture(scope="session")
def fred_data_dir() -> Path:
    return PROJECT_ROOT / "fred_fed_data"
