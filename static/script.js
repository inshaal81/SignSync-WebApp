const video = document.querySelector('#videoElement');
const canvas = document.querySelector('#canvasElement');
const capturedImage = document.querySelector('#capturedImage');
const videoPlaceholder = document.querySelector('#videoPlaceholder');
const imagePlaceholder = document.querySelector('#imagePlaceholder');
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
        const imageDataUrl = canvas.toDataURL('image/jpeg'); // or 'image/png'
        
        // Display the captured image and hide placeholder
        capturedImage.src = imageDataUrl;
        imagePlaceholder.style.display = 'none';
        capturedImage.style.display = 'flex';

        // The imageDataUrl can then be sent to a server if needed
        console.log("Captured image data URL:", imageDataUrl.substring(0, 50) + "...");
    } else {
        alert("Webcam not started.");
    }
}
