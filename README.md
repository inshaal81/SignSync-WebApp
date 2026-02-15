# SignSync - Web App

A real-time American Sign Language (ASL) classification web application powered by machine learning.

![Python](https://img.shields.io/badge/Python-3.11+-blue.svg)
![Flask](https://img.shields.io/badge/Flask-2.3.0+-green.svg)
![License](https://img.shields.io/badge/License-MIT-yellow.svg)

SignSync enables users to capture hand gestures via webcam and receive instant ASL letter predictions with confidence scores. Built with Flask and designed for easy deployment on Render.

---

## Table of Contents

- [Features](#features)
- [Tech Stack](#tech-stack)
- [Project Structure](#project-structure)
- [Local Development](#local-development)
- [Environment Variables](#environment-variables)
- [Deployment on Render](#deployment-on-render)
- [API Endpoints](#api-endpoints)
- [Adding Your ML Model](#adding-your-ml-model)
- [Security Considerations](#security-considerations)
- [Testing & CI/CD](#testing--cicd)
- [Troubleshooting](#troubleshooting)
- [License](#license)

---

## Features

- **Real-time Webcam Capture** - Start, stop, and capture snapshots directly from your browser
- **ASL Letter Classification** - Identify hand gestures for all 26 letters of the ASL alphabet
- **Confidence Scores** - Each prediction includes a confidence percentage
- **Interactive Documentation** - Built-in "How It Works" page explaining the technology
- **Production-Ready** - Configured for Render deployment with Gunicorn
- **Secure by Default** - Enforces SECRET_KEY in production, validates file uploads
- **Automatic Cleanup** - Temporary uploaded images are deleted immediately after classification

---

## Tech Stack

| Layer | Technology |
|-------|------------|
| **Backend** | Flask 2.3.0+, Python 3.11+ |
| **Image Processing** | Pillow 10.0.0+ |
| **WSGI Server** | Gunicorn 21.0.0+ |
| **Environment** | python-dotenv 1.0.0+ |
| **Frontend** | HTML5, CSS3, Vanilla JavaScript |
| **Deployment** | Render (render.yaml configured) |

---

## Project Structure

```
SignSync-WebApp/
├── app.py                 # Flask application with routes and classification
├── requirements.txt       # Python dependencies
├── render.yaml            # Render deployment configuration
├── .env.example           # Environment variable template
├── .gitignore             # Git ignore rules
├── README.md              # This file
├── template/
│   ├── index.html         # Main page with webcam interface
│   └── docs.html          # "How It Works" documentation page
├── static/
│   ├── script.js          # Webcam capture and classification logic
│   ├── style.css          # Main page styles
│   └── docStyle.css       # Documentation page styles
├── model/                 # Directory for ML model files (add your own)
└── uploads/               # Temporary image storage (auto-cleaned)
```

---

## Local Development

### Prerequisites

- Python 3.11 or higher
- pip (Python package manager)
- A modern web browser with webcam support

### Setup

1. **Clone the repository**

   ```bash
   git clone https://github.com/theChosen-1/SignSync-WebApp.git
   cd SignSync-WebApp
   ```

2. **Create and activate a virtual environment**

   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**

   ```bash
   pip install -r requirements.txt
   ```

4. **Set up environment variables**

   ```bash
   cp .env.example .env
   # Edit .env with your preferred settings (optional for development)
   ```

5. **Run the development server**

   ```bash
   python app.py
   ```

6. **Open your browser**

   Navigate to [http://localhost:5001](http://localhost:5001)

---

## Environment Variables

| Variable | Description | Required | Default |
|----------|-------------|----------|---------|
| `SECRET_KEY` | Flask secret key for session security | Yes (production) | `dev-only-not-for-production` |
| `FLASK_DEBUG` | Enable debug mode (`True`/`False`) | No | `False` |
| `FLASK_ENV` | Environment (`development`/`production`) | No | Inferred from DEBUG |
| `PORT` | Server port number | No | `5001` (local), `10000` (Render) |
| `MODEL_PATH` | Path to ML model file | No | `model/signsync_model.pkl` |
| `MODEL_TYPE` | Model type (`custom`/`sklearn`/`tensorflow`/`onnx`) | No | `custom` |
| `MODEL_INPUT_SIZE` | Expected input image size (pixels) | No | `28` |
| `MAX_CONTENT_LENGTH` | Max upload size in bytes | No | `16777216` (16MB) |
| `LOG_LEVEL` | Logging level | No | `INFO` |

### Generating a SECRET_KEY

For production, generate a secure secret key:

```bash
python -c "import secrets; print(secrets.token_hex(32))"
```

---

## Deployment on Render

### Automatic Deployment

1. **Push your code to GitHub**

2. **Create a new Web Service on [Render](https://render.com)**

3. **Connect your GitHub repository**

4. **Configure environment variables**

   In Render's dashboard, add:
   - `SECRET_KEY`: Your generated secret key (required)
   - `FLASK_DEBUG`: `False` (recommended for production)

5. **Deploy**

   Render will automatically detect `render.yaml` and configure:
   - Python 3.11 runtime
   - Build command: `pip install -r requirements.txt`
   - Start command: `gunicorn --bind 0.0.0.0:$PORT --workers 4 --threads 2 --timeout 120 app:app`
   - Health check path: `/health`
   - Auto-deploy on push: enabled
   - Auto-generated `SECRET_KEY`

Your app will be live at `https://signsync.onrender.com` (or your custom domain).

### Manual Configuration

If not using `render.yaml`, configure these settings manually:

- **Runtime**: Python 3.11
- **Build Command**: `pip install -r requirements.txt`
- **Start Command**: `gunicorn --bind 0.0.0.0:$PORT app:app`

---

## API Endpoints

### GET `/`

Returns the main page with webcam interface.

**Response**: HTML page

---

### GET `/docs`

Returns the "How It Works" documentation page.

**Response**: HTML page

---

### GET `/health`

Health check endpoint for monitoring and deployment verification.

**Response**:
```json
{
  "status": "healthy",
  "timestamp": "2024-01-15T12:30:45.123456",
  "version": "1.0.0",
  "checks": {
    "model_loaded": true,
    "model_type": "custom",
    "upload_directory": true,
    "disk_space_ok": true
  }
}
```

**Status Codes**:
- `200 OK` - All critical checks passed
- `503 Service Unavailable` - One or more critical checks failed (status: "degraded")

---

### POST `/classify`

Classifies an ASL hand gesture from an uploaded image.

**Request**:
- Content-Type: `multipart/form-data`
- Body: `image` (file) - Image file (PNG, JPG, JPEG, or GIF)

**Successful Response** (200):
```json
{
  "success": true,
  "classification": "A",
  "confidence": 0.95
}
```

**Error Responses**:

| Status | Response | Cause |
|--------|----------|-------|
| 400 | `{"error": "No image file provided"}` | Missing `image` field |
| 400 | `{"error": "No file selected"}` | Empty filename |
| 400 | `{"error": "Invalid file type. Only images are allowed."}` | Unsupported file extension |
| 500 | `{"error": "Classification error: ..."}` | Model/processing failure |
| 500 | `{"error": "Server error: ..."}` | General server error |

**Example using curl**:

```bash
curl -X POST -F "image=@hand_sign.jpg" http://localhost:5001/classify
```

---

## Adding Your ML Model

The application includes placeholder classification that returns random letters. To integrate your trained ASL model:

### Step 1: Add Your Model File

Place your trained model in the `model/` directory:

```
model/
├── asl_model.h5      # TensorFlow/Keras
├── asl_model.pt      # PyTorch
└── asl_model.pkl     # Scikit-learn
```

### Step 2: Update Dependencies

Add your ML framework to `requirements.txt`:

```
# For TensorFlow
tensorflow>=2.13.0

# For PyTorch
torch>=2.0.0
torchvision>=0.15.0

# For Scikit-learn
scikit-learn>=1.3.0
```

### Step 3: Modify app.py

Update the model loading and classification function:

```python
# MODEL LOADING
# Replace this section in app.py

import tensorflow as tf  # or import torch

# Load your model
MODEL = tf.keras.models.load_model('model/asl_model.h5')

def classify_asl_image(image_path):
    """
    ASL classification function.
    """
    from PIL import Image
    import numpy as np

    # Load and preprocess image
    img = Image.open(image_path).convert('RGB')
    img = img.resize((224, 224))  # Adjust to your model's input size
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    # Run prediction
    predictions = MODEL.predict(img_array)
    predicted_index = np.argmax(predictions[0])
    confidence = float(predictions[0][predicted_index])

    # Map index to letter
    letters = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'
    classification = letters[predicted_index]

    return classification, confidence
```

---

## Security Considerations

### Production Requirements

- **SECRET_KEY**: The application will raise an error if `SECRET_KEY` is not set in production mode. Never use the default development key in production.

### File Upload Security

- **Allowed Extensions**: Only `png`, `jpg`, `jpeg`, and `gif` files are accepted
- **File Size Limit**: Maximum upload size is 16MB
- **Automatic Cleanup**: Uploaded files are deleted immediately after classification

### Debug Mode

- Set `FLASK_DEBUG=False` in production
- Debug mode exposes detailed error messages that could reveal sensitive information

### Recommendations

1. Always use HTTPS in production (Render provides this automatically)
2. Regularly rotate your SECRET_KEY
3. Monitor the `/health` endpoint for uptime
4. Review logs for suspicious classification requests

---

## Testing & CI/CD

### Running Tests

The project includes a comprehensive test suite using pytest:

```bash
# Run all tests
pytest

# Run with coverage report
pytest --cov=app --cov-report=term-missing

# Run specific test file
pytest tests/test_app.py
```

### Test Coverage

Current test coverage: **69%** with 50 passing tests covering:
- Utility functions (`allowed_file`, activation functions)
- All Flask routes (`/`, `/docs`, `/health`, `/classify`)
- Image preprocessing pipeline
- Neural network forward pass

### Continuous Integration

GitHub Actions automatically runs on every push and pull request:
- **Test Job**: Runs pytest with coverage
- **Lint Job**: Checks code formatting with Black, isort, and Flake8

View workflow status: Check the "Actions" tab in the GitHub repository.

### Code Formatting

```bash
# Install development tools
pip install black isort flake8

# Format code
black app.py tests/
isort app.py tests/

# Check formatting
black --check app.py tests/
flake8 app.py tests/
```

---

## Troubleshooting

### Webcam Not Working

- **Browser Permissions**: Ensure you've granted camera access when prompted
- **HTTPS Required**: Some browsers require HTTPS for webcam access (not an issue on localhost)
- **Device Availability**: Check if another application is using the camera

### Classification Errors

- **Image Quality**: Ensure good lighting and a plain background
- **Hand Position**: Keep your entire hand visible in frame
- **File Size**: Images larger than 16MB will be rejected

### Deployment Issues

- **SECRET_KEY Missing**: Add `SECRET_KEY` environment variable in Render dashboard
- **Build Failures**: Check that `requirements.txt` is properly formatted
- **Port Binding**: Ensure no other service is using port 10000 on Render

### Development Mode

To enable detailed error messages locally:

```bash
FLASK_DEBUG=True python app.py
```

---

## License

MIT License

Copyright (c) 2024 theChosen-1

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
