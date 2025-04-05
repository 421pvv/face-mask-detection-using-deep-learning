"""
Face Mask Detection Web Application

This Flask application provides a web interface for face mask detection
using webcam or uploaded images. It uses a TensorFlow-based model to
classify faces as:
- With mask
- Without mask
- Mask worn incorrectly

The app is optimized for macOS, particularly Apple Silicon devices.
"""
import os
import cv2
import numpy as np
from pathlib import Path
from flask import Flask, render_template, Response, request, jsonify
import logging
import time

# Import our detection utilities
from detection_utils import initialize_detectors, process_image

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Initialize the Flask app
app = Flask(__name__)

# Global variables
camera = None

# Initialize detectors
try:
    face_detector, mask_detector = initialize_detectors()
    logger.info("Detection models loaded successfully")
except Exception as e:
    logger.error(f"Error loading detection models: {e}")
    face_detector, mask_detector = None, None


def get_camera():
    """
    Get or initialize the camera device

    Returns:
        Camera object or None if initialization fails
    """
    global camera
    if camera is None:
        try:
            camera = cv2.VideoCapture(0)
            # Set resolution to 640x480 for better performance
            camera.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

            # Check if camera opened successfully
            if not camera.isOpened():
                logger.error("Failed to open camera")
                raise RuntimeError(
                    "Could not open camera. Please check your webcam connection.")
        except Exception as e:
            logger.error(f"Camera initialization error: {e}")
            return None

    return camera


def release_camera():
    """Release the camera resources"""
    global camera
    if camera is not None:
        camera.release()
        camera = None
        logger.info("Camera resources released")


def generate_frames():
    """
    Generator function that yields video frames with mask detection

    This function captures frames from the webcam, processes them for
    face mask detection, and yields them as JPEG images for streaming.

    Yields:
        JPEG frames for streaming through HTTP
    """
    # Ensure detectors are initialized
    if face_detector is None or mask_detector is None:
        logger.error("Detection models not initialized")
        return

    # Initialize camera
    try:
        camera = get_camera()

        # Check if camera is working
        if not camera or not camera.isOpened():
            logger.error("Camera failed to open")
            return

        logger.info("Camera initialized successfully for streaming")
    except Exception as e:
        logger.error(f"Camera error: {e}")
        return

    frame_count = 0
    error_count = 0

    while True:
        try:
            # Read frame
            success, frame = camera.read()

            if not success:
                error_count += 1
                logger.error(
                    f"Failed to capture frame from camera (error {error_count})")

                # If we have too many consecutive errors, break the loop
                if error_count > 5:
                    logger.error(
                        "Too many consecutive frame capture errors, stopping stream")
                    break

                # Yield a blank frame with error message
                blank_frame = np.zeros((480, 640, 3), np.uint8)
                cv2.putText(blank_frame, "Camera Error", (200, 240),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

                ret, buffer = cv2.imencode('.jpg', blank_frame)
                if ret:
                    frame_bytes = buffer.tobytes()
                    yield (b'--frame\r\n'
                           b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

                # Wait a moment before trying again
                time.sleep(0.5)
                continue

            # Reset error count on successful frame capture
            error_count = 0
            frame_count += 1

            # Skip frames occasionally to reduce processing load
            if frame_count % 3 != 0:  # Process every 3rd frame
                continue

            # Add frame counter in bottom corner
            height, width = frame.shape[:2]
            cv2.putText(frame, f"Frame: {frame_count}", (10, height - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

            # Process the frame with mask detection
            processed_frame, detections = process_image(
                frame, face_detector, mask_detector)

            # Log detection results periodically
            if frame_count % 30 == 0:  # Log every 30th processed frame
                logger.info(
                    f"Frame {frame_count}: Detected {len(detections)} faces")

            # Convert to JPEG
            ret, buffer = cv2.imencode('.jpg', processed_frame)
            if not ret:
                logger.error("Failed to encode processed frame")
                # Fallback to original frame
                ret, buffer = cv2.imencode('.jpg', frame)
                if not ret:
                    logger.error("Failed to encode original frame as fallback")
                    continue

            # Yield the frame in byte format
            frame_bytes = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

        except Exception as e:
            logger.error(f"Error in frame generator: {e}")

            # If an error occurs, return a blank frame with error message
            try:
                blank_frame = np.zeros((480, 640, 3), np.uint8)
                cv2.putText(blank_frame, "Processing Error", (180, 240),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

                ret, buffer = cv2.imencode('.jpg', blank_frame)
                if ret:
                    frame_bytes = buffer.tobytes()
                    yield (b'--frame\r\n'
                           b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
            except Exception as inner_e:
                logger.error(f"Failed to create error frame: {inner_e}")
                # Just wait a moment before continuing
                time.sleep(0.1)


@app.route('/')
def index():
    """Home page route"""
    return render_template('index.html')


@app.route('/livestream')
def livestream():
    """Livestream page route"""
    return render_template('livestream.html')


@app.route('/video_feed')
def video_feed():
    """Video streaming route for webcam feed"""
    return Response(
        generate_frames(),
        mimetype='multipart/x-mixed-replace; boundary=frame'
    )


@app.route('/stop_stream')
def stop_stream():
    """Stop the video stream and release camera resources"""
    release_camera()
    return {'status': 'success', 'message': 'Camera released'}


@app.route('/api/detect', methods=['POST'])
def api_detect():
    """API endpoint for mask detection from uploaded image"""
    try:
        # Check if detectors are initialized
        if face_detector is None or mask_detector is None:
            logger.error("Detection models not initialized")
            return jsonify({'error': 'Detection models not initialized'}), 500

        # Check if an image was uploaded
        if 'image' not in request.files:
            logger.error("No image provided in upload request")
            return jsonify({'error': 'No image provided'}), 400

        # Get the uploaded file
        file = request.files['image']
        logger.info(f"Processing uploaded image: {file.filename}")

        # Read the image bytes
        img_bytes = file.read()
        if not img_bytes:
            logger.error("Uploaded file is empty")
            return jsonify({'error': 'Uploaded file is empty'}), 400

        # Convert bytes to numpy array
        nparr = np.frombuffer(img_bytes, np.uint8)
        logger.debug(f"Image bytes converted to array of shape {nparr.shape}")

        # Decode the image
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if img is None:
            logger.error("Failed to decode image")
            return jsonify({'error': 'Invalid image format'}), 400

        logger.debug(f"Decoded image shape: {img.shape}")

        # Process the image
        logger.info("Processing image for face mask detection")
        processed_img, detections = process_image(
            img, face_detector, mask_detector)

        logger.info(f"Detection complete: found {len(detections)} faces")

        # Convert back to JPEG
        _, img_encoded = cv2.imencode('.jpg', processed_img)
        if not _:
            logger.error("Failed to encode processed image")
            return jsonify({'error': 'Failed to encode result image'}), 500

        response = img_encoded.tobytes()

        logger.info("Returning processed image")
        return Response(response, mimetype='image/jpeg')

    except Exception as e:
        logger.error(f"Unexpected error processing uploaded image: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return jsonify({'error': str(e)}), 500


@app.route('/about')
def about():
    """About page route"""
    return render_template('about.html')


@app.errorhandler(500)
def server_error(e):
    """Handle internal server errors"""
    logger.error(f"Server error: {e}")
    return render_template('error.html', error=str(e)), 500


# Cleanup when the app is stopped
@app.teardown_appcontext
def cleanup(exception=None):
    """Clean up resources when app context ends"""
    release_camera()


if __name__ == '__main__':
    # Create directories if they don't exist
    os.makedirs('models/face_detector', exist_ok=True)

    # Check if models are available before starting the app
    if not Path('models/face_mask_detection_model').exists():
        logger.warning(
            "Face mask detection model not found in 'models/face_mask_detection_model'")

    # Ignore resource tracker warnings
    import warnings
    warnings.filterwarnings("ignore", category=UserWarning,
                            module="multiprocessing.resource_tracker")

    # Start the Flask app
    port = int(os.environ.get('PORT', 3000))
    app.run(host='0.0.0.0', port=port, debug=True, threaded=True)
