"""
conftest.py - Pytest Configuration
==================================
Shared fixtures and configuration for tests.
"""

import pytest
import tempfile
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))


@pytest.fixture(scope="session")
def project_root():
    """Return project root directory."""
    return Path(__file__).parent.parent


@pytest.fixture(scope="session")
def temp_data_dir():
    """Create a session-scoped temporary data directory."""
    with tempfile.TemporaryDirectory() as tmpdir:
        # Create some test files
        data_path = Path(tmpdir)
        
        # Test text file
        (data_path / "test.txt").write_text(
            "Tunisia is a country in North Africa. "
            "It has a rich history dating back to ancient Carthage.",
            encoding="utf-8"
        )
        
        # Test Arabic file
        (data_path / "arabic.txt").write_text(
            "تونس بلد في شمال أفريقيا. لها تاريخ غني يعود إلى قرطاج القديمة.",
            encoding="utf-8"
        )
        
        # Test French file
        (data_path / "french.txt").write_text(
            "La Tunisie est un pays d'Afrique du Nord. "
            "Elle a une riche histoire remontant à l'ancienne Carthage.",
            encoding="utf-8"
        )
        
        yield tmpdir


@pytest.fixture
def sample_chunks():
    """Create sample chunks for testing."""
    from src.data.chunking import Chunk
    
    return [
        Chunk(
            content="Tunisia gained independence in 1956.",
            chunk_id="chunk_1",
            doc_id="doc_1",
            chunk_index=0,
            metadata={"source": "history.txt", "language": "en"}
        ),
        Chunk(
            content="Couscous is a traditional Tunisian dish.",
            chunk_id="chunk_2",
            doc_id="doc_1",
            chunk_index=1,
            metadata={"source": "cuisine.txt", "language": "en"}
        ),
        Chunk(
            content="The Medina of Tunis is a UNESCO World Heritage Site.",
            chunk_id="chunk_3",
            doc_id="doc_2",
            chunk_index=0,
            metadata={"source": "culture.txt", "language": "en"}
        )
    ]


@pytest.fixture
def sample_documents():
    """Create sample documents for testing."""
    from src.data.ingestion import Document
    
    return [
        Document(
            content="Tunisia is a North African country with a Mediterranean coast.",
            source="geography.txt",
            doc_type="txt",
            metadata={"topic": "geography"}
        ),
        Document(
            content="The Tunisian Revolution of 2010-2011 led to democratic reforms.",
            source="history.txt",
            doc_type="txt",
            metadata={"topic": "history"}
        )
    ]


# Configure pytest
def pytest_configure(config):
    """Configure pytest."""
    config.addinivalue_line(
        "markers", "slow: marks tests as slow (deselect with '-m \"not slow\"')"
    )
    config.addinivalue_line(
        "markers", "integration: marks tests as integration tests"
    )


# Skip slow tests by default unless --runslow is given
def pytest_addoption(parser):
    parser.addoption(
        "--runslow", action="store_true", default=False, help="run slow tests"
    )
    parser.addoption(
        "--runintegration", action="store_true", default=False, help="run integration tests"
    )


def pytest_collection_modifyitems(config, items):
    if not config.getoption("--runslow"):
        skip_slow = pytest.mark.skip(reason="need --runslow option to run")
        for item in items:
            if "slow" in item.keywords:
                item.add_marker(skip_slow)
    
    if not config.getoption("--runintegration"):
        skip_integration = pytest.mark.skip(reason="need --runintegration option to run")
        for item in items:
            if "integration" in item.keywords:
                item.add_marker(skip_integration)
