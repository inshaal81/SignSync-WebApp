from flask import Flask, render_template, request, jsonify
import os
import sys
import logging
import shutil
from datetime import datetime
from werkzeug.utils import secure_filename
from dotenv import load_dotenv
from PIL import Image
import numpy as np
import random
import string

# Load environment variables from .env file
load_dotenv()

app = Flask(__name__, template_folder='template')

# =============================================================================
# CONFIGURATION
# =============================================================================
UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size

# Production settings
DEBUG = os.environ.get('FLASK_DEBUG', 'False').lower() == 'true'

# SECRET_KEY configuration - must be set in production
secret_key = os.environ.get('SECRET_KEY')
if not secret_key and not DEBUG:
    raise ValueError('SECRET_KEY environment variable must be set in production')
app.config['SECRET_KEY'] = secret_key or 'dev-only-not-for-production'

# Model configuration
MODEL_PATH = os.environ.get('MODEL_PATH', 'model/signsync_model.pkl')
MODEL_TYPE = os.environ.get('MODEL_TYPE', 'custom')
MODEL_INPUT_SIZE = int(os.environ.get('MODEL_INPUT_SIZE', '28'))

# ASL alphabet mapping (A-Z, excluding J and Z which require motion)
ASL_LABELS = list('ABCDEFGHIKLMNOPQRSTUVWXY')

# =============================================================================
# LOGGING
# =============================================================================
logging.basicConfig(
    level=logging.DEBUG if DEBUG else logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Ensure uploads directory exists
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# =============================================================================
# MODEL LOADING
# =============================================================================
MODEL = None

def load_model():
    """
    Load the ML model with error handling.

    Supports multiple model types:
    - custom: Custom neural network stored as dict with parameters
    - sklearn: scikit-learn models (pickle files)
    - tensorflow: TensorFlow/Keras models (.h5 files)
    - onnx: ONNX models (.onnx files)

    Returns:
        Loaded model object or None if loading fails
    """
    global MODEL

    if not os.path.exists(MODEL_PATH):
        logger.warning(f'Model file not found at {MODEL_PATH}. Using placeholder classification.')
        return None

    try:
        if MODEL_TYPE == 'custom':
            import pickle
            with open(MODEL_PATH, 'rb') as f:
                MODEL = pickle.load(f)
            logger.info(f'Custom neural network loaded from {MODEL_PATH}')
            logger.info(f'Model architecture: {MODEL.get("layer_dims", "unknown")}')
            logger.info(f'Model test accuracy: {MODEL.get("test_accuracy", "unknown")}')

        elif MODEL_TYPE == 'sklearn':
            import pickle
            with open(MODEL_PATH, 'rb') as f:
                MODEL = pickle.load(f)
            logger.info(f'Scikit-learn model loaded successfully from {MODEL_PATH}')

        elif MODEL_TYPE == 'tensorflow':
            import tensorflow as tf
            MODEL = tf.keras.models.load_model(MODEL_PATH)
            logger.info(f'TensorFlow model loaded successfully from {MODEL_PATH}')

        elif MODEL_TYPE == 'onnx':
            import onnxruntime as ort
            MODEL = ort.InferenceSession(MODEL_PATH)
            logger.info(f'ONNX model loaded successfully from {MODEL_PATH}')

        else:
            logger.error(f'Unknown model type: {MODEL_TYPE}')
            return None

        return MODEL

    except Exception as e:
        logger.error(f'Failed to load model: {str(e)}', exc_info=True)
        if not DEBUG:
            logger.warning('Model failed to load in production mode - using placeholder')
        return None

# Load model at startup
MODEL = load_model()

# =============================================================================
# CUSTOM NEURAL NETWORK FORWARD PASS
# =============================================================================
def sigmoid(z):
    """Sigmoid activation function."""
    return 1 / (1 + np.exp(-np.clip(z, -500, 500)))

def relu(z):
    """ReLU activation function."""
    return np.maximum(0, z)

def softmax(z):
    """Softmax activation function for output layer."""
    exp_z = np.exp(z - np.max(z, axis=0, keepdims=True))
    return exp_z / np.sum(exp_z, axis=0, keepdims=True)

def forward_pass(X, parameters):
    """
    Forward pass through the custom neural network.

    Args:
        X: Input data, shape (n_features, n_samples)
        parameters: Dictionary containing weights and biases

    Returns:
        tuple: (output probabilities, cache for debugging)
    """
    cache = {}
    A = X

    # Determine number of layers from parameters
    L = len(parameters) // 2

    # Hidden layers with ReLU activation
    for l in range(1, L):
        W = parameters[f'W{l}']
        b = parameters[f'b{l}']
        Z = np.dot(W, A) + b
        A = relu(Z)
        cache[f'A{l}'] = A

    # Output layer with softmax activation
    W = parameters[f'W{L}']
    b = parameters[f'b{L}']
    Z = np.dot(W, A) + b
    A = softmax(Z)
    cache[f'A{L}'] = A

    return A, cache

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# =============================================================================
# IMAGE PREPROCESSING
# =============================================================================
def preprocess_image(image_path, target_size=None):
    """
    Preprocess image for model input.

    Args:
        image_path: Path to image file
        target_size: Target dimensions (width, height). Defaults to MODEL_INPUT_SIZE.

    Returns:
        numpy.ndarray: Preprocessed image array ready for model input
    """
    if target_size is None:
        target_size = (MODEL_INPUT_SIZE, MODEL_INPUT_SIZE)

    try:
        img = Image.open(image_path)

        # Convert to grayscale (ASL models typically use grayscale)
        if img.mode != 'L':
            img = img.convert('L')

        # Resize to model input size
        img = img.resize(target_size, Image.Resampling.LANCZOS)

        # Convert to numpy array
        img_array = np.array(img)

        # Normalize pixel values to 0-1 range
        img_array = img_array.astype('float32') / 255.0

        # Reshape based on model type
        if MODEL_TYPE == 'custom':
            # Custom NN expects (n_features, n_samples) - column vector
            img_array = img_array.flatten().reshape(-1, 1)
        elif MODEL_TYPE == 'sklearn':
            # Sklearn expects (n_samples, n_features) - row vector
            img_array = img_array.flatten().reshape(1, -1)
        else:
            # CNN models expect (batch, height, width, channels)
            img_array = np.expand_dims(img_array, axis=(0, -1))

        return img_array

    except Exception as e:
        logger.error(f'Image preprocessing error: {str(e)}', exc_info=True)
        raise

def classify_asl_image(image_path):
    """
    ASL classification function.

    1. Load and preprocess image (resize to model's expected input size)
    2. Run prediction through MODEL
    3. Return letter and confidence score

    Args:
        image_path: Path to the image file

    Returns:
        tuple: (classification, confidence) - e.g., ('A', 0.95)
    """
    if MODEL is None:
        # Placeholder: Return random letter with random confidence
        logger.debug('Using placeholder classification (no model loaded)')
        random_letter = random.choice(ASL_LABELS)
        confidence = round(random.uniform(0.7, 0.99), 2)
        return random_letter, confidence

    try:
        # Preprocess the image
        processed_image = preprocess_image(image_path)

        if MODEL_TYPE == 'custom':
            # Custom neural network prediction
            parameters = MODEL['parameters']
            predictions, _ = forward_pass(processed_image, parameters)
            predicted_class = np.argmax(predictions)
            confidence = float(predictions[predicted_class, 0])

        elif MODEL_TYPE == 'sklearn':
            # Scikit-learn prediction
            predictions = MODEL.predict_proba(processed_image)
            predicted_class = np.argmax(predictions[0])
            confidence = float(predictions[0][predicted_class])

        elif MODEL_TYPE == 'tensorflow':
            # TensorFlow/Keras prediction
            predictions = MODEL.predict(processed_image, verbose=0)
            predicted_class = np.argmax(predictions[0])
            confidence = float(predictions[0][predicted_class])

        elif MODEL_TYPE == 'onnx':
            # ONNX prediction
            input_name = MODEL.get_inputs()[0].name
            predictions = MODEL.run(None, {input_name: processed_image})
            predicted_class = np.argmax(predictions[0][0])
            confidence = float(predictions[0][0][predicted_class])

        else:
            raise ValueError(f'Unknown model type: {MODEL_TYPE}')

        # Map class index to ASL letter
        classification = ASL_LABELS[predicted_class] if predicted_class < len(ASL_LABELS) else '?'

        return classification, round(confidence, 2)

    except Exception as e:
        logger.error(f'Model prediction error: {str(e)}', exc_info=True)
        raise

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/docs')
def docs():
    return render_template('docs.html')

@app.route('/health')
def health():
    """Detailed health check endpoint."""
    def check_disk_space():
        """Check if sufficient disk space is available."""
        try:
            total, used, free = shutil.disk_usage(UPLOAD_FOLDER)
            return free > 100 * 1024 * 1024  # True if more than 100MB free
        except Exception:
            return False

    health_status = {
        'status': 'healthy',
        'timestamp': datetime.now().isoformat(),
        'version': '1.0.0',
        'checks': {
            'model_loaded': MODEL is not None,
            'model_type': MODEL_TYPE if MODEL is not None else None,
            'upload_directory': os.path.exists(UPLOAD_FOLDER) and os.access(UPLOAD_FOLDER, os.W_OK),
            'disk_space_ok': check_disk_space()
        }
    }

    # Determine overall status
    critical_checks = [
        health_status['checks']['upload_directory'],
        health_status['checks']['disk_space_ok']
    ]
    all_critical_pass = all(critical_checks)
    health_status['status'] = 'healthy' if all_critical_pass else 'degraded'

    status_code = 200 if all_critical_pass else 503
    return jsonify(health_status), status_code

@app.route('/classify', methods=['POST'])
def classify():
    try:
        # Check if file is present in request
        if 'image' not in request.files:
            logger.warning('Classification request received without image file')
            return jsonify({'error': 'No image file provided'}), 400

        file = request.files['image']

        # Check if file is selected
        if file.filename == '':
            logger.warning('Classification request received with empty filename')
            return jsonify({'error': 'No file selected'}), 400

        # Validate file type
        if not allowed_file(file.filename):
            logger.warning(f'Invalid file type attempted: {file.filename}')
            return jsonify({'error': 'Invalid file type. Only images are allowed.'}), 400

        # Generate unique filename with timestamp
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f'asl_{timestamp}.jpg'
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)

        # Save file temporarily
        file.save(filepath)
        logger.info(f'Image saved temporarily: {filename}')

        try:
            # Classify the image
            classification, confidence = classify_asl_image(filepath)
            logger.info(f'Classification result: {classification} (confidence: {confidence})')

            # Delete the image file immediately after classification
            if os.path.exists(filepath):
                os.remove(filepath)
                logger.debug(f'Temporary file deleted: {filename}')

            # Return classification result
            return jsonify({
                'classification': classification,
                'confidence': confidence,
                'success': True
            })

        except Exception as e:
            logger.error(f'Classification error: {str(e)}', exc_info=True)
            # Ensure file is deleted even if classification fails
            if os.path.exists(filepath):
                os.remove(filepath)
            return jsonify({'error': f'Classification error: {str(e)}'}), 500

    except Exception as e:
        logger.error(f'Server error: {str(e)}', exc_info=True)
        return jsonify({'error': f'Server error: {str(e)}'}), 500

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5001))
    app.run(host='0.0.0.0', port=port, debug=DEBUG)
