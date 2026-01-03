const video = document.querySelector('#videoElement');
const canvas = document.querySelector('#canvasElement');
const capturedImage = document.querySelector('#capturedImage');
const videoPlaceholder = document.querySelector('#videoPlaceholder');
const imagePlaceholder = document.querySelector('#imagePlaceholder');
const resultDiv = document.querySelector('.result');
const context = canvas.getContext('2d');
let webcamStream = null;

function startWebcam() {
    if (navigator.mediaDevices && navigator.mediaDevices.getUserMedia) {
        // Request access to the video stream
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
                };
            })
            .catch(function(err) {
                console.log("An error occurred: " + err);
                alert("Could not access the webcam. Please ensure permissions are granted.");
            });
    } else {
        alert("getUserMedia is not supported by your browser.");
    }
}

function stopWebcam() {
    if (webcamStream) {
        // Stop all tracks in the stream (both video and audio if applicable)
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

        // Send image to Flask backend for classification
        sendImageForClassification(imageDataUrl);
    } else {
        alert("Webcam not started.");
    }
}

async function sendImageForClassification(imageDataUrl) {
    try {
        // Show loading state
        resultDiv.innerHTML = '<div class="loading">Processing...</div>';
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
        
        if (result.success) {
            // Display classification result
            resultDiv.innerHTML = `
                <div class="classification-result">
                    <div class="classification-letter">${result.classification}</div>
                    <div class="confidence">Confidence: ${(result.confidence * 100).toFixed(1)}%</div>
                </div>
            `;
        } else {
            throw new Error(result.error || 'Classification failed');
        }
        
    } catch (error) {
        console.error('Error classifying image:', error);
        resultDiv.innerHTML = `<div class="error">Error: ${error.message}</div>`;
    }
}
