const video = document.querySelector('#videoElement');
const canvas = document.querySelector('#canvasElement');
const capturedImage = document.querySelector('#capturedImage');
const videoPlaceholder = document.querySelector('#videoPlaceholder');
const imagePlaceholder = document.querySelector('#imagePlaceholder');
const resultDiv = document.querySelector('.result');
const context = canvas.getContext('2d');
let webcamStream = null;

// Button references for loading states
const startBtn = document.querySelector('[onclick="startWebcam()"]');
const snapshotBtn = document.querySelector('[onclick="takeSnapshot()"]');
const stopBtn = document.querySelector('[onclick="stopWebcam()"]');

// Button loading state helpers
function setButtonLoading(button, loading, originalText) {
    if (!button) return;
    if (loading) {
        button.disabled = true;
        button.dataset.originalText = button.textContent;
        button.textContent = 'Loading...';
        button.classList.add('loading');
    } else {
        button.disabled = false;
        button.textContent = originalText || button.dataset.originalText || button.textContent;
        button.classList.remove('loading');
    }
}

// Browser compatibility check
function checkBrowserCompatibility() {
    if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
        // Disable webcam buttons
        if (startBtn) startBtn.disabled = true;
        if (snapshotBtn) snapshotBtn.disabled = true;
        if (stopBtn) stopBtn.disabled = true;
        return false;
    }
    return true;
}

function startWebcam() {
    if (!checkBrowserCompatibility()) return;

    setButtonLoading(startBtn, true);

    navigator.mediaDevices.getUserMedia({ video: true, audio: false })
        .then(function(stream) {
            webcamStream = stream;
            video.srcObject = stream;
            // Hide placeholder and show video
            videoPlaceholder.style.display = 'none';
            video.style.display = 'flex';
            video.onloadedmetadata = () => {
                // Set canvas dimensions to match video feed for a proper capture
                canvas.width = video.videoWidth;
                canvas.height = video.videoHeight;
                setButtonLoading(startBtn, false, 'Start Webcam');
            };
        })
        .catch(function(err) {
            console.error('Webcam error:', err);
            setButtonLoading(startBtn, false, 'Start Webcam');

            let message = 'Could not access webcam.';
            if (err.name === 'NotAllowedError') {
                message = 'Camera access denied. Please allow camera permissions in your browser settings.';
            } else if (err.name === 'NotFoundError') {
                message = 'No camera found. Please connect a webcam and try again.';
            } else if (err.name === 'NotReadableError') {
                message = 'Camera is in use by another application.';
            }

            resultDiv.innerHTML = `<div class="error">${message}</div>`;
            resultDiv.classList.add('has-result');
        });
}

function stopWebcam() {
    if (webcamStream) {
        // Stop all tracks in the stream
        webcamStream.getTracks().forEach(track => track.stop());
        video.srcObject = null;
        webcamStream = null;
        // Show placeholder and hide video
        video.style.display = 'none';
        videoPlaceholder.style.display = 'flex';
    }
}

function takeSnapshot() {
    if (webcamStream) {
        // Draw the current video frame onto the canvas
        context.drawImage(video, 0, 0, canvas.width, canvas.height);

        // Convert the canvas content to a data URL (base64 image)
        const imageDataUrl = canvas.toDataURL('image/jpeg');

        // Display the captured image and hide placeholder
        capturedImage.src = imageDataUrl;
        imagePlaceholder.style.display = 'none';
        capturedImage.style.display = 'flex';

        // Show coming soon message
        resultDiv.innerHTML = `
            <div class="classification-result">
                <div class="classification-letter">Coming Soon</div>
                <div class="confidence">MediaPipe + LSTM model in development</div>
                <div class="processing-time">Real-time ASL recognition</div>
            </div>
        `;
        resultDiv.classList.add('has-result');
    } else {
        resultDiv.innerHTML = '<div class="error">Please start the webcam first</div>';
        resultDiv.classList.add('has-result');
    }
}

// Initialize on page load
document.addEventListener('DOMContentLoaded', function() {
    checkBrowserCompatibility();
});
