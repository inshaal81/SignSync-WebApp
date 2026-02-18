"""
Unit tests for SignSync-WebApp Flask application.
"""
import os
import sys

import pytest

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


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
        assert json_data['status'] == 'healthy'

    def test_health_has_version_field(self, client):
        """Test that health response contains version field."""
        response = client.get('/health')
        json_data = response.get_json()
        assert 'version' in json_data
        assert json_data['version'] == '2.0.0'
