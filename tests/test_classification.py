"""
Tests for classification pipeline and neural network functions.
"""
import os
import sys
import tempfile

import pytest
import numpy as np
from PIL import Image

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app import sigmoid, relu, softmax, forward_pass, preprocess_image, classify_asl_image


class TestSigmoid:
    """Tests for the sigmoid activation function."""

    def test_sigmoid_zero(self):
        """Test sigmoid(0) = 0.5."""
        result = sigmoid(0)
        assert np.isclose(result, 0.5)

    def test_sigmoid_positive(self):
        """Test sigmoid of positive value is > 0.5."""
        result = sigmoid(5)
        assert result > 0.5
        assert result < 1.0

    def test_sigmoid_negative(self):
        """Test sigmoid of negative value is < 0.5."""
        result = sigmoid(-5)
        assert result < 0.5
        assert result > 0.0

    def test_sigmoid_large_positive(self):
        """Test sigmoid approaches 1 for large positive values."""
        result = sigmoid(100)
        assert np.isclose(result, 1.0, atol=1e-6)

    def test_sigmoid_large_negative(self):
        """Test sigmoid approaches 0 for large negative values."""
        result = sigmoid(-100)
        assert np.isclose(result, 0.0, atol=1e-6)

    def test_sigmoid_array(self):
        """Test sigmoid works with numpy arrays."""
        arr = np.array([-1, 0, 1])
        result = sigmoid(arr)
        assert result.shape == arr.shape
        assert np.all(result > 0)
        assert np.all(result < 1)


class TestRelu:
    """Tests for the ReLU activation function."""

    def test_relu_positive(self):
        """Test ReLU of positive value returns the value."""
        assert relu(5) == 5
        assert relu(0.5) == 0.5

    def test_relu_zero(self):
        """Test ReLU of zero returns zero."""
        assert relu(0) == 0

    def test_relu_negative(self):
        """Test ReLU of negative value returns zero."""
        assert relu(-5) == 0
        assert relu(-0.5) == 0

    def test_relu_array(self):
        """Test ReLU works with numpy arrays."""
        arr = np.array([-2, -1, 0, 1, 2])
        result = relu(arr)
        expected = np.array([0, 0, 0, 1, 2])
        np.testing.assert_array_equal(result, expected)


class TestSoftmax:
    """Tests for the softmax activation function."""

    def test_softmax_sum_to_one(self):
        """Test softmax outputs sum to 1."""
        z = np.array([[1], [2], [3]])
        result = softmax(z)
        assert np.isclose(np.sum(result), 1.0)

    def test_softmax_probabilities(self):
        """Test softmax outputs are valid probabilities."""
        z = np.array([[1], [2], [3]])
        result = softmax(z)
        assert np.all(result >= 0)
        assert np.all(result <= 1)

    def test_softmax_larger_value_higher_probability(self):
        """Test that larger input values have higher probability."""
        z = np.array([[1], [5], [2]])
        result = softmax(z)
        assert result[1, 0] > result[0, 0]
        assert result[1, 0] > result[2, 0]

    def test_softmax_equal_inputs(self):
        """Test softmax of equal inputs gives equal probabilities."""
        z = np.array([[1], [1], [1]])
        result = softmax(z)
        expected = np.array([[1/3], [1/3], [1/3]])
        np.testing.assert_array_almost_equal(result, expected)


class TestForwardPass:
    """Tests for the neural network forward pass."""

    def test_forward_pass_single_layer(self):
        """Test forward pass with a simple single hidden layer network."""
        # Input: 4 features, 1 sample
        X = np.array([[1], [2], [3], [4]])

        # Parameters for 4 -> 3 -> 2 network
        parameters = {
            'W1': np.random.randn(3, 4) * 0.01,
            'b1': np.zeros((3, 1)),
            'W2': np.random.randn(2, 3) * 0.01,
            'b2': np.zeros((2, 1))
        }

        output, cache = forward_pass(X, parameters)

        # Check output shape
        assert output.shape == (2, 1)
        # Check output is valid probabilities (softmax)
        assert np.isclose(np.sum(output), 1.0)
        assert np.all(output >= 0)
        assert np.all(output <= 1)

    def test_forward_pass_multi_layer(self):
        """Test forward pass with multiple hidden layers."""
        # Input: 784 features (28x28 image), 1 sample
        X = np.random.randn(784, 1)

        # Parameters for 784 -> 128 -> 64 -> 24 network
        parameters = {
            'W1': np.random.randn(128, 784) * 0.01,
            'b1': np.zeros((128, 1)),
            'W2': np.random.randn(64, 128) * 0.01,
            'b2': np.zeros((64, 1)),
            'W3': np.random.randn(24, 64) * 0.01,
            'b3': np.zeros((24, 1))
        }

        output, cache = forward_pass(X, parameters)

        # Check output shape (24 classes for ASL)
        assert output.shape == (24, 1)
        # Check output is valid probabilities
        assert np.isclose(np.sum(output), 1.0)

    def test_forward_pass_cache_populated(self):
        """Test that forward pass populates the cache correctly."""
        X = np.random.randn(4, 1)
        parameters = {
            'W1': np.random.randn(3, 4) * 0.01,
            'b1': np.zeros((3, 1)),
            'W2': np.random.randn(2, 3) * 0.01,
            'b2': np.zeros((2, 1))
        }

        output, cache = forward_pass(X, parameters)

        assert 'A1' in cache
        assert 'A2' in cache


class TestPreprocessImage:
    """Tests for the image preprocessing function."""

    @pytest.fixture
    def grayscale_image(self, temp_dir):
        """Create a grayscale test image."""
        img_path = os.path.join(temp_dir, 'gray_test.png')
        img_array = np.random.randint(0, 255, (50, 50), dtype=np.uint8)
        img = Image.fromarray(img_array, mode='L')
        img.save(img_path)
        return img_path

    @pytest.fixture
    def color_image(self, temp_dir):
        """Create an RGB color test image."""
        img_path = os.path.join(temp_dir, 'color_test.png')
        img_array = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
        img = Image.fromarray(img_array, mode='RGB')
        img.save(img_path)
        return img_path

    def test_preprocess_grayscale_conversion(self, color_image):
        """Test that color images are converted to grayscale."""
        result = preprocess_image(color_image)
        # For custom model type, result is flattened (28*28=784 features)
        assert result.shape[0] == 784 or result.shape[1] == 784

    def test_preprocess_resize(self, grayscale_image):
        """Test that images are resized correctly."""
        result = preprocess_image(grayscale_image, target_size=(28, 28))
        # For custom model, shape should be (784, 1)
        assert result.shape == (784, 1)

    def test_preprocess_normalization(self, grayscale_image):
        """Test that pixel values are normalized to 0-1 range."""
        result = preprocess_image(grayscale_image)
        assert np.all(result >= 0.0)
        assert np.all(result <= 1.0)

    def test_preprocess_custom_target_size(self, grayscale_image):
        """Test preprocessing with custom target size."""
        result = preprocess_image(grayscale_image, target_size=(32, 32))
        # For custom model, shape should be (32*32, 1) = (1024, 1)
        assert result.shape == (1024, 1)


class TestClassifyAslImage:
    """Tests for the ASL classification function."""

    def test_classify_returns_tuple(self, test_image):
        """Test that classify returns a tuple of (letter, confidence)."""
        result = classify_asl_image(test_image)
        assert isinstance(result, tuple)
        assert len(result) == 2

    def test_classify_returns_valid_letter(self, test_image):
        """Test that classification returns a valid ASL letter."""
        asl_labels = list('ABCDEFGHIKLMNOPQRSTUVWXY')
        classification, confidence = classify_asl_image(test_image)
        assert classification in asl_labels

    def test_classify_returns_valid_confidence(self, test_image):
        """Test that classification returns a valid confidence score."""
        classification, confidence = classify_asl_image(test_image)
        assert isinstance(confidence, float)
        assert 0.0 <= confidence <= 1.0

    def test_classify_different_images(self, temp_dir):
        """Test that classification handles different images."""
        # Create multiple test images
        results = []
        for i in range(3):
            img_path = os.path.join(temp_dir, f'test_{i}.png')
            img_array = np.random.randint(0, 255, (28, 28), dtype=np.uint8)
            img = Image.fromarray(img_array, mode='L')
            img.save(img_path)
            result = classify_asl_image(img_path)
            results.append(result)

        # All results should be valid
        asl_labels = list('ABCDEFGHIKLMNOPQRSTUVWXY')
        for classification, confidence in results:
            assert classification in asl_labels
            assert 0.0 <= confidence <= 1.0
