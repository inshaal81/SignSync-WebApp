from flask import Flask, render_template, request, jsonify
import os
from datetime import datetime
from werkzeug.utils import secure_filename
import random
import string

app = Flask(__name__, template_folder='template')

# CONFIG
UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size

# Production settings
DEBUG = os.environ.get('FLASK_DEBUG', 'False').lower() == 'true'

# Ensure uploads directory exists
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# MODEL LOADING 
MODEL = None  # (using placeholder classification)

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

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

    if MODEL is not None:
        # Add actual model prediction here
        pass

    # Placeholder: Return a random letter A-Z with random confidence
    random_letter = random.choice(string.ascii_uppercase)
    confidence = round(random.uniform(0.7, 0.99), 2)

    return random_letter, confidence

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/health')
def health():
    return jsonify({'status': 'healthy'}), 200

@app.route('/classify', methods=['POST'])
def classify():
    try:
        # Check if file is present in request
        if 'image' not in request.files:
            return jsonify({'error': 'No image file provided'}), 400
        
        file = request.files['image']
        
        # Check if file is selected
        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400
        
        # Validate file type
        if not allowed_file(file.filename):
            return jsonify({'error': 'Invalid file type. Only images are allowed.'}), 400
        
        # Generate unique filename with timestamp
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f'asl_{timestamp}.jpg'
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        
        # Save file temporarily
        file.save(filepath)
        
        try:
            # Classify the image
            classification, confidence = classify_asl_image(filepath)
            
            # Delete the image file immediately after classification
            if os.path.exists(filepath):
                os.remove(filepath)
            
            # Return classification result
            return jsonify({
                'classification': classification,
                'confidence': confidence,
                'success': True
            })
        
        except Exception as e:
            # Ensure file is deleted even if classification fails
            if os.path.exists(filepath):
                os.remove(filepath)
            return jsonify({'error': f'Classification error: {str(e)}'}), 500
    
    except Exception as e:
        return jsonify({'error': f'Server error: {str(e)}'}), 500

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5001))
    app.run(host='0.0.0.0', port=port, debug=DEBUG)
