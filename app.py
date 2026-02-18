from flask import Flask, render_template, jsonify
import os
import logging
from datetime import datetime
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

app = Flask(__name__, template_folder='template')

# =============================================================================
# CONFIGURATION
# =============================================================================

# Production settings
DEBUG = os.environ.get('FLASK_DEBUG', 'False').lower() == 'true'

# SECRET_KEY configuration - must be set in production
secret_key = os.environ.get('SECRET_KEY')
if not secret_key and not DEBUG:
    raise ValueError('SECRET_KEY environment variable must be set in production')
app.config['SECRET_KEY'] = secret_key or 'dev-only-not-for-production'

# =============================================================================
# LOGGING
# =============================================================================
logging.basicConfig(
    level=logging.DEBUG if DEBUG else logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# =============================================================================
# ROUTES
# =============================================================================

@app.route('/')
def index():
    return render_template('index.html')


@app.route('/docs')
def docs():
    return render_template('docs.html')


@app.route('/health')
def health():
    """Health check endpoint for Render deployment."""
    health_status = {
        'status': 'healthy',
        'timestamp': datetime.now().isoformat(),
        'version': '2.0.0',
        'message': 'Frontend ready - ML model coming soon'
    }
    return jsonify(health_status), 200


if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5001))
    app.run(host='0.0.0.0', port=port, debug=DEBUG)
