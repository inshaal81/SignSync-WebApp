"""
Pytest fixtures for SignSync-WebApp tests.
"""
import os
import sys
import tempfile
import shutil

# Set test environment variables BEFORE any app imports
os.environ['FLASK_DEBUG'] = 'True'
os.environ['SECRET_KEY'] = 'test-secret-key-for-testing'

import pytest
from PIL import Image
import numpy as np

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


@pytest.fixture
def temp_dir():
    """Create a temporary directory for test files."""
    temp_path = tempfile.mkdtemp()
    yield temp_path
    # Cleanup after test
    shutil.rmtree(temp_path, ignore_errors=True)


@pytest.fixture
def test_image(temp_dir):
    """Create a test image file (grayscale, 28x28)."""
    img_path = os.path.join(temp_dir, 'test_image.png')
    # Create a simple grayscale test image
    img_array = np.random.randint(0, 255, (28, 28), dtype=np.uint8)
    img = Image.fromarray(img_array, mode='L')
    img.save(img_path)
    yield img_path


@pytest.fixture
def test_color_image(temp_dir):
    """Create a test RGB color image file."""
    img_path = os.path.join(temp_dir, 'test_color_image.jpg')
    # Create a simple RGB test image
    img_array = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
    img = Image.fromarray(img_array, mode='RGB')
    img.save(img_path)
    yield img_path


@pytest.fixture
def invalid_file(temp_dir):
    """Create an invalid (non-image) file."""
    file_path = os.path.join(temp_dir, 'invalid.txt')
    with open(file_path, 'w') as f:
        f.write('This is not an image file.')
    yield file_path
