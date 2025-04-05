"""
Face Mask Detection Utilities

This module provides classes and functions for detecting faces and classifying mask usage,
handling both image and video processing for the face mask detection application.

The module contains:
1. FaceDetector - A class for face detection using either OpenCV DNN or Haar Cascade
2. MaskDetector - A class for face mask classification using a TensorFlow model
3. process_image - A function that combines detection and classification

Optimized for Apple Silicon Macs with fallback options for different hardware.
"""
import os
import cv2
import numpy as np
import tensorflow as tf
from pathlib import Path
import logging

# Configure module logger
logger = logging.getLogger(__name__)


class FaceDetector:
    """Face detection using either OpenCV's DNN or Haar Cascade detector.

    This class provides face detection functionality with automatic fallback
    between different detection methods based on what's available and works
    best on the current hardware.
    """

    def __init__(self, confidence_threshold=0.5):
        """Initialize the face detector

        Args:
            confidence_threshold: Minimum confidence threshold for DNN detection (0.0-1.0)
        """
        self.confidence_threshold = confidence_threshold
        self.detection_method = None

        # Try loading the Haar cascade first (more reliable on M1/M2 Macs)
        try:
            face_cascade_path = cv2.data.haarcascades + \
                'haarcascade_frontalface_default.xml'

            if os.path.exists(face_cascade_path):
                self.model = cv2.CascadeClassifier(face_cascade_path)

                if not self.model.empty():
                    self.detection_method = "haar"
                    logger.info(
                        f"Loaded Haar Cascade face detector from {face_cascade_path}")
                else:
                    logger.warning(
                        "Failed to load Haar Cascade model (empty model)")
            else:
                logger.warning(
                    f"Haar Cascade file not found at {face_cascade_path}")
        except Exception as e:
            logger.error(f"Could not load Haar Cascade face detector: {e}")

        # If Haar Cascade failed, try DNN
        if self.detection_method is None:
            try:
                # Use DNN face detector (more accurate but may have issues on Apple Silicon)
                prototxt_path = Path('models/face_detector/deploy.prototxt')
                weights_path = Path(
                    'models/face_detector/res10_300x300_ssd_iter_140000.caffemodel')

                # Create models directory if it doesn't exist
                os.makedirs('models/face_detector', exist_ok=True)

                # Download model files if they don't exist
                if not prototxt_path.exists() or not weights_path.exists():
                    self._download_face_detector()

                self.model = cv2.dnn.readNet(
                    str(prototxt_path), str(weights_path))
                self.detection_method = "dnn"
                logger.info(f"Loaded DNN face detector from {weights_path}")
            except Exception as e:
                logger.error(f"Could not load DNN face detector: {e}")
                raise ValueError("Failed to initialize any face detector")

    def detect_faces(self, image):
        """Detect faces in an image

        Args:
            image: Input image (numpy array)

        Returns:
            List of face bounding boxes as (x, y, w, h)
        """
        if self.detection_method == "dnn":
            return self._detect_faces_dnn(image)
        else:
            return self._detect_faces_haar(image)

    def _detect_faces_dnn(self, image):
        """Detect faces using OpenCV DNN face detector

        Args:
            image: Input image

        Returns:
            List of face bounding boxes as (x, y, w, h)
        """
        (h, w) = image.shape[:2]

        # Create a blob from the image
        blob = cv2.dnn.blobFromImage(
            cv2.resize(image, (300, 300)),
            1.0,
            (300, 300),
            (104.0, 177.0, 123.0)
        )

        # Pass the blob through the network
        self.model.setInput(blob)
        detections = self.model.forward()

        # Initialize the list of faces
        faces = []

        # Loop over the detections
        for i in range(0, detections.shape[2]):
            # Extract the confidence
            confidence = detections[0, 0, i, 2]

            # Filter weak detections
            if confidence > self.confidence_threshold:
                # Compute the (x, y)-coordinates of the bounding box
                box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                (startX, startY, endX, endY) = box.astype("int")

                # Ensure the bounding boxes fall within the dimensions of the frame
                startX, startY = max(0, startX), max(0, startY)
                endX, endY = min(w - 1, endX), min(h - 1, endY)

                # Convert from (startX, startY, endX, endY) to (x, y, w, h)
                x, y = startX, startY
                w, h = endX - startX, endY - startY

                # Only add faces with valid dimensions
                if w > 0 and h > 0:
                    faces.append((x, y, w, h))

        return faces

    def _detect_faces_haar(self, image):
        """Detect faces using Haar cascade classifier

        Args:
            image: Input image

        Returns:
            List of face bounding boxes as (x, y, w, h)
        """
        # Convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(
            image.shape) == 3 else image

        # Ensure gray image is valid
        if gray is None or gray.size == 0:
            logger.warning("Invalid grayscale image for face detection")
            return []

        logger.debug(f"Running Haar detection on image of shape {gray.shape}")

        # Try different parameters to ensure detection works
        faces = self.model.detectMultiScale(
            gray,
            scaleFactor=1.2,  # Less strict scale factor
            minNeighbors=4,   # Fewer neighbors required
            minSize=(30, 30)
        )

        # If no faces detected, try with more permissive parameters
        if len(faces) == 0:
            logger.debug(
                "No faces detected with initial parameters, trying more permissive settings")
            faces = self.model.detectMultiScale(
                gray,
                scaleFactor=1.3,
                minNeighbors=3,
                minSize=(20, 20)
            )

        # Convert the output to list of tuples
        face_list = [tuple(face) for face in faces] if len(faces) > 0 else []

        logger.debug(f"Detected {len(face_list)} faces with Haar cascade")
        return face_list

    def _download_face_detector(self):
        """Download the OpenCV face detector model files"""
        import urllib.request

        # Create output directory
        os.makedirs('models/face_detector', exist_ok=True)

        # Define file paths
        prototxt_path = Path('models/face_detector/deploy.prototxt')
        weights_path = Path(
            'models/face_detector/res10_300x300_ssd_iter_140000.caffemodel')

        # Download files if they don't exist
        if not prototxt_path.exists():
            prototxt_url = "https://raw.githubusercontent.com/opencv/opencv/master/samples/dnn/face_detector/deploy.prototxt"
            logger.info(f"Downloading {prototxt_url} to {prototxt_path}")
            urllib.request.urlretrieve(prototxt_url, prototxt_path)

        if not weights_path.exists():
            weights_url = "https://raw.githubusercontent.com/opencv/opencv_3rdparty/dnn_samples_face_detector_20170830/res10_300x300_ssd_iter_140000.caffemodel"
            logger.info(f"Downloading {weights_url} to {weights_path}")
            urllib.request.urlretrieve(weights_url, weights_path)


class MaskDetector:
    """Face mask detection model wrapper

    This class loads and uses a trained TensorFlow model to detect
    whether a person is wearing a mask correctly, incorrectly, or not at all.
    """

    def __init__(self, model_path='models/face_mask_detection_model', img_height=224, img_width=224):
        """Initialize the mask detector

        Args:
            model_path: Path to the saved TensorFlow model
            img_height: Input image height expected by the model
            img_width: Input image width expected by the model
        """
        self.img_height = img_height
        self.img_width = img_width
        self.model = None

        # Default class mapping
        self.class_names = {
            0: 'with_mask',
            1: 'without_mask',
            2: 'mask_weared_incorrect'
        }

        # Load class names from file if available
        class_indices_path = Path('models/class_indices.txt')
        if class_indices_path.exists():
            self.class_names = {}
            with open(class_indices_path, 'r') as f:
                for line in f:
                    name, idx = line.strip().split(',')
                    self.class_names[int(idx)] = name

        # Load the model
        try:
            self.model = tf.saved_model.load(model_path)
            logger.info(f"Loaded mask detection model from {model_path}")
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            raise

    def predict(self, face_img):
        """Predict mask status for a face image

        Args:
            face_img: Face image array

        Returns:
            predicted_class: Class name (with_mask, without_mask, mask_weared_incorrect)
            confidence: Prediction confidence
            class_idx: Class index
        """
        if self.model is None:
            raise ValueError("Model not loaded")

        # Preprocess the face
        preprocessed = self._preprocess_face(face_img)

        try:
            # Handle different TensorFlow model serving approaches
            if hasattr(self.model, 'predict'):
                predictions = self.model.predict(preprocessed)
            elif hasattr(self.model, 'signatures'):
                infer = self.model.signatures['serving_default']
                predictions = infer(tf.constant(preprocessed))

                # Get the output tensor name
                output_key = list(predictions.keys())[0]
                predictions = predictions[output_key].numpy()
            else:
                # Direct call (newest TF versions)
                predictions = self.model(preprocessed).numpy()

            # Ensure predictions is a numpy array
            predictions = np.array(predictions)

            # Handle multi-dimensional predictions
            if len(predictions.shape) > 1:
                predictions = predictions[0]

            # Flatten to 1D array if needed
            predictions = predictions.flatten()

            # Get class index and confidence
            class_idx = int(np.argmax(predictions))
            confidence = float(predictions[class_idx])

            # Get class name
            class_name = self.class_names.get(class_idx, f"Class {class_idx}")

            return class_name, confidence, class_idx

        except Exception as e:
            logger.error(f"Error making prediction: {e}")
            logger.debug(f"Preprocessed shape: {preprocessed.shape}")
            if 'predictions' in locals():
                logger.debug(
                    f"Predictions shape: {predictions.shape if isinstance(predictions, np.ndarray) else 'not a numpy array'}")
                logger.debug(f"Predictions: {predictions}")
            raise

    def _preprocess_face(self, face_img):
        """Preprocess a face image for the model

        Args:
            face_img: Face image array

        Returns:
            Preprocessed face tensor
        """
        # Resize image
        face_img = cv2.resize(face_img, (self.img_height, self.img_width))

        # Convert BGR to RGB if needed
        if len(face_img.shape) == 3 and face_img.shape[2] == 3:
            face_img = cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB)

        # Add batch dimension and normalize
        face_img = np.expand_dims(face_img, axis=0).astype(np.float32) / 255.0

        return face_img


def process_image(image, face_detector, mask_detector):
    """Process an image for face mask detection

    This function takes an input image, detects faces, and then classifies
    each face for mask usage.

    Args:
        image: Input image
        face_detector: FaceDetector instance
        mask_detector: MaskDetector instance

    Returns:
        processed_image: Image with bounding boxes and labels
        detection_results: List of detection dictionaries with 'box', 'class', and 'confidence'
    """
    # Colors for different mask statuses (BGR format)
    color_map = {
        'with_mask': (0, 255, 0),           # Green
        'without_mask': (0, 0, 255),        # Red
        'mask_weared_incorrect': (0, 165, 255)  # Orange
    }

    # Make a copy for drawing
    output_image = image.copy()

    # Store detection results
    detection_results = []

    logger.debug(f"Processing image shape: {image.shape}")

    try:
        # Detect faces
        faces = face_detector.detect_faces(image)
        logger.debug(f"Detected {len(faces)} faces")

        # Process each detected face
        for i, (x, y, w, h) in enumerate(faces):
            # Ensure coordinates are valid
            if x < 0 or y < 0 or w <= 0 or h <= 0 or x + w > image.shape[1] or y + h > image.shape[0]:
                logger.warning(
                    f"Invalid face coordinates: x={x}, y={y}, w={w}, h={h}, image shape={image.shape}")
                continue

            # Extract face ROI (Region of Interest)
            face_roi = image[y:y+h, x:x+w]

            # Skip empty ROIs
            if face_roi.size == 0:
                logger.warning(f"Empty face ROI for face #{i+1}")
                continue

            logger.debug(
                f"Processing face #{i+1}, ROI shape: {face_roi.shape}")

            try:
                # Get prediction
                class_name, confidence, _ = mask_detector.predict(face_roi)

                # Get color based on prediction
                color = color_map.get(class_name, (255, 255, 255))

                # Draw bounding box - thicker line for better visibility
                cv2.rectangle(output_image, (x, y), (x + w, y + h), color, 3)

                # Prepare label text
                label = f"{class_name}: {confidence:.2f}"

                # Get text size for background
                text_size, _ = cv2.getTextSize(
                    label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
                text_w, text_h = text_size

                # Ensure text background stays within image boundaries
                bg_y1 = max(0, y - text_h - 10)
                bg_y2 = y

                # Draw label background
                cv2.rectangle(output_image, (x, bg_y1),
                              (x + text_w + 10, bg_y2), color, -1)

                # Draw label text (white)
                cv2.putText(output_image, label, (x + 5, bg_y2 - 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

                # Add to results
                detection_results.append({
                    'box': (x, y, w, h),
                    'class': class_name,
                    'confidence': confidence
                })

                logger.debug(
                    f"Face #{i+1} classified as {class_name} with confidence {confidence:.2f}")

            except Exception as e:
                logger.error(f"Error processing face #{i+1}: {e}")

    except Exception as e:
        logger.error(f"Error during face detection: {e}")

    return output_image, detection_results


def initialize_detectors():
    """Initialize both face and mask detectors

    Returns:
        face_detector: Initialized FaceDetector instance
        mask_detector: Initialized MaskDetector instance
    """
    try:
        # Initialize face detector
        face_detector = FaceDetector(confidence_threshold=0.5)

        # Initialize mask detector
        mask_detector = MaskDetector()

        return face_detector, mask_detector

    except Exception as e:
        logger.error(f"Error initializing detectors: {e}")
        raise
