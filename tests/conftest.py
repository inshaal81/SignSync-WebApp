"""
Pytest fixtures for SignSync-WebApp tests.
"""
import os
import sys

# Set test environment variables BEFORE any app imports
os.environ['FLASK_DEBUG'] = 'True'
os.environ['SECRET_KEY'] = 'test-secret-key-for-testing'

import pytest

# Add project root to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


@pytest.fixture
def app():
    """Create and configure a test application instance."""
    from app import app as flask_app

    flask_app.config.update({
        'TESTING': True,
        'WTF_CSRF_ENABLED': False,
    })

    yield flask_app


@pytest.fixture
def client(app):
    """Create a test client for the application."""
    return app.test_client()
