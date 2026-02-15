"""
Unit tests for SignSync-WebApp Flask application.
"""
import io
import os
import sys

import pytest
from PIL import Image
import numpy as np

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app import allowed_file, ALLOWED_EXTENSIONS


class TestAllowedFile:
    """Tests for the allowed_file utility function."""

    def test_allowed_file_valid_png(self):
        """Test that PNG files are allowed."""
        assert allowed_file('image.png') is True

    def test_allowed_file_valid_jpg(self):
        """Test that JPG files are allowed."""
        assert allowed_file('image.jpg') is True

    def test_allowed_file_valid_jpeg(self):
        """Test that JPEG files are allowed."""
        assert allowed_file('image.jpeg') is True

    def test_allowed_file_valid_gif(self):
        """Test that GIF files are allowed."""
        assert allowed_file('image.gif') is True

    def test_allowed_file_invalid_txt(self):
        """Test that TXT files are not allowed."""
        assert allowed_file('document.txt') is False

    def test_allowed_file_invalid_pdf(self):
        """Test that PDF files are not allowed."""
        assert allowed_file('document.pdf') is False

    def test_allowed_file_no_extension(self):
        """Test that files without extension are not allowed."""
        assert allowed_file('noextension') is False

    def test_allowed_file_empty_string(self):
        """Test that empty string is not allowed."""
        assert allowed_file('') is False

    def test_allowed_file_uppercase(self):
        """Test that uppercase extensions are allowed (case insensitive)."""
        assert allowed_file('image.PNG') is True
        assert allowed_file('image.JPG') is True

    def test_allowed_file_mixed_case(self):
        """Test that mixed case extensions are allowed."""
        assert allowed_file('image.JpG') is True


class TestIndexRoute:
    """Tests for the index route."""

    def test_index_returns_200(self, client):
        """Test that index route returns 200 status code."""
        response = client.get('/')
        assert response.status_code == 200

    def test_index_returns_html(self, client):
        """Test that index route returns HTML content."""
        response = client.get('/')
        assert b'<!DOCTYPE html>' in response.data or b'<html' in response.data


class TestDocsRoute:
    """Tests for the docs route."""

    def test_docs_returns_200(self, client):
        """Test that docs route returns 200 status code."""
        response = client.get('/docs')
        assert response.status_code == 200

    def test_docs_returns_html(self, client):
        """Test that docs route returns HTML content."""
        response = client.get('/docs')
        assert b'<!DOCTYPE html>' in response.data or b'<html' in response.data


class TestHealthRoute:
    """Tests for the health check route."""

    def test_health_returns_200(self, client):
        """Test that health route returns 200 status code."""
        response = client.get('/health')
        assert response.status_code == 200

    def test_health_returns_json(self, client):
        """Test that health route returns JSON content."""
        response = client.get('/health')
        assert response.content_type == 'application/json'

    def test_health_has_status_field(self, client):
        """Test that health response contains status field."""
        response = client.get('/health')
        json_data = response.get_json()
        assert 'status' in json_data
        assert json_data['status'] in ['healthy', 'degraded']

    def test_health_has_checks_field(self, client):
        """Test that health response contains checks field."""
        response = client.get('/health')
        json_data = response.get_json()
        assert 'checks' in json_data
        assert 'upload_directory' in json_data['checks']
        assert 'disk_space_ok' in json_data['checks']


class TestClassifyRoute:
    """Tests for the classify route."""

    def test_classify_no_file(self, client):
        """Test classify route returns 400 when no file is provided."""
        response = client.post('/classify')
        assert response.status_code == 400
        json_data = response.get_json()
        assert 'error' in json_data
        assert 'No image file provided' in json_data['error']

    def test_classify_empty_filename(self, client):
        """Test classify route returns 400 when filename is empty."""
        data = {'image': (io.BytesIO(b''), '')}
        response = client.post('/classify', data=data, content_type='multipart/form-data')
        assert response.status_code == 400
        json_data = response.get_json()
        assert 'error' in json_data

    def test_classify_invalid_file_type(self, client, invalid_file):
        """Test classify route returns 400 for invalid file type."""
        with open(invalid_file, 'rb') as f:
            data = {'image': (f, 'test.txt')}
            response = client.post('/classify', data=data, content_type='multipart/form-data')
        assert response.status_code == 400
        json_data = response.get_json()
        assert 'error' in json_data
        assert 'Invalid file type' in json_data['error']

    def test_classify_valid_image(self, client, test_image):
        """Test classify route returns 200 for valid image."""
        with open(test_image, 'rb') as f:
            data = {'image': (f, 'test.png')}
            response = client.post('/classify', data=data, content_type='multipart/form-data')
        assert response.status_code == 200
        json_data = response.get_json()
        assert 'classification' in json_data
        assert 'confidence' in json_data
        assert 'success' in json_data
        assert json_data['success'] is True

    def test_classify_valid_jpg_image(self, client, test_color_image):
        """Test classify route handles JPG images."""
        with open(test_color_image, 'rb') as f:
            data = {'image': (f, 'test.jpg')}
            response = client.post('/classify', data=data, content_type='multipart/form-data')
        assert response.status_code == 200
        json_data = response.get_json()
        assert json_data['success'] is True

    def test_classify_returns_valid_asl_letter(self, client, test_image):
        """Test that classification returns a valid ASL letter."""
        asl_labels = list('ABCDEFGHIKLMNOPQRSTUVWXY')
        with open(test_image, 'rb') as f:
            data = {'image': (f, 'test.png')}
            response = client.post('/classify', data=data, content_type='multipart/form-data')
        json_data = response.get_json()
        assert json_data['classification'] in asl_labels

    def test_classify_returns_valid_confidence(self, client, test_image):
        """Test that classification returns a valid confidence score."""
        with open(test_image, 'rb') as f:
            data = {'image': (f, 'test.png')}
            response = client.post('/classify', data=data, content_type='multipart/form-data')
        json_data = response.get_json()
        assert 0.0 <= json_data['confidence'] <= 1.0
