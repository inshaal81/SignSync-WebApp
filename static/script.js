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

// Notification system
function showNotification(message, type = 'info') {
    // Remove existing notification if any
    const existing = document.querySelector('.notification');
    if (existing) existing.remove();

    const notification = document.createElement('div');
    notification.className = `notification notification-${type}`;
    notification.textContent = message;
    document.body.appendChild(notification);

    // Trigger animation
    requestAnimationFrame(() => {
        notification.classList.add('show');
    });

    // Auto-remove after 4 seconds
    setTimeout(() => {
        notification.classList.remove('show');
        setTimeout(() => notification.remove(), 300);
    }, 4000);
}

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
        showNotification(
            'Your browser does not support webcam access. Please use Chrome, Firefox, or Safari.',
            'error'
        );
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
                showNotification('Webcam started successfully', 'success');
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

            showNotification(message, 'error');
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
        showNotification('Webcam stopped', 'info');
    }
}

function takeSnapshot() {
    if (webcamStream) {
        setButtonLoading(snapshotBtn, true);

        // Draw the current video frame onto the canvas
        context.drawImage(video, 0, 0, canvas.width, canvas.height);

        // Convert the canvas content to a data URL (base64 image)
        const imageDataUrl = canvas.toDataURL('image/jpeg');

        // Display the captured image and hide placeholder
        capturedImage.src = imageDataUrl;
        imagePlaceholder.style.display = 'none';
        capturedImage.style.display = 'flex';

        // Send image to Flask backend for classification
        sendImageForClassification(imageDataUrl);
    } else {
        showNotification('Please start the webcam first', 'error');
    }
}

async function sendImageForClassification(imageDataUrl) {
    const startTime = performance.now();

    try {
        // Show loading state
        resultDiv.innerHTML = '<div class="loading">Classifying...</div>';
        resultDiv.style.display = 'flex';

        // Convert data URL to Blob
        const base64Data = imageDataUrl.split(',')[1];
        const byteCharacters = atob(base64Data);
        const byteNumbers = new Array(byteCharacters.length);
        for (let i = 0; i < byteCharacters.length; i++) {
            byteNumbers[i] = byteCharacters.charCodeAt(i);
        }
        const byteArray = new Uint8Array(byteNumbers);
        const blob = new Blob([byteArray], { type: 'image/jpeg' });

        // Create FormData and append the image
        const formData = new FormData();
        formData.append('image', blob, 'capture.jpg');

        // Send to Flask backend
        const classifyResponse = await fetch('/classify', {
            method: 'POST',
            body: formData
        });

        if (!classifyResponse.ok) {
            const errorData = await classifyResponse.json().catch(() => ({}));
            throw new Error(errorData.error || `Server error: ${classifyResponse.status}`);
        }

        const result = await classifyResponse.json();
        const processingTime = ((performance.now() - startTime) / 1000).toFixed(2);

        if (result.success) {
            // Display classification result with processing time
            resultDiv.innerHTML = `
                <div class="classification-result">
                    <div class="classification-letter">${result.classification}</div>
                    <div class="confidence">Confidence: ${(result.confidence * 100).toFixed(1)}%</div>
                    <div class="processing-time">Processed in ${processingTime}s</div>
                </div>
            `;
            showNotification(`Detected letter: ${result.classification}`, 'success');
        } else {
            throw new Error(result.error || 'Classification failed');
        }

    } catch (error) {
        console.error('Error classifying image:', error);
        resultDiv.innerHTML = `<div class="error">Error: ${error.message}</div>`;
        showNotification(`Classification failed: ${error.message}`, 'error');
    } finally {
        setButtonLoading(snapshotBtn, false, 'Take Snapshot');
    }
}

// Initialize on page load
document.addEventListener('DOMContentLoaded', function() {
    checkBrowserCompatibility();
});
