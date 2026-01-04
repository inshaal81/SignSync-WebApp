# SignSync - Web App

A web application for real-time American Sign Language (ASL) classification using machine learning.

## Features

- Webcam capture for live hand gesture recognition
- Image upload support
- Real-time ASL letter classification with confidence scores

## Local Development

### Prerequisites

- Python 3.11+

### Setup

1. Clone the repository:
   ```bash
   git clone https://github.com/theChosen-1/SignSync-WebApp.git
   cd SignSync-WebApp
   ```

2. Create a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

4. Run the development server:
   ```bash
   python app.py
   ```

5. Open http://localhost:5001 in your browser.

## Deployment on Render

1. Push your code to GitHub.

2. Go to [Render](https://render.com) and create a new Web Service.

3. Connect your GitHub repository.

4. Render will automatically detect the `render.yaml` configuration.

5. Deploy!

Your app will be live at `https://signsync.onrender.com` (or similar).

## Adding Your Model

The app currently uses placeholder classification. To add your trained model:

1. Place your model file in the `model/` directory.

2. Update `app.py`:
   - Import your ML framework (TensorFlow, PyTorch, etc.)
   - Load your model in the MODEL LOADING section
   - Update `classify_asl_image()` with actual prediction logic

See the comments in `app.py` for examples.

## Project Structure

```
SignSync-WebApp/
├── app.py              # Flask backend
├── requirements.txt    # Python dependencies
├── render.yaml         # Render deployment config
├── template/
│   └── index.html      # Main page
├── static/
│   ├── script.js       # Frontend logic
│   └── style.css       # Styling
├── model/              # Your model files (add here)
└── uploads/            # Temporary image storage
```

## License

MIT
