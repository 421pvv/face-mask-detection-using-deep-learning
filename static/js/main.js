/**
 * Face Mask Detection - Main JavaScript File
 *
 * This file contains all the JavaScript functionality for the Face Mask Detection app,
 * including form handling, image upload, and UI interactions.
 */

document.addEventListener('DOMContentLoaded', function () {
  console.log('Face Mask Detection App Initialized');

  // Initialize all components
  initializeFileInputs();
  initializeWebcamHandling();
  initializeUploadForm();
});

/**
 * Enhances file input styling and functionality
 */
function initializeFileInputs() {
  const fileInputs = document.querySelectorAll('input[type="file"]');

  fileInputs.forEach((input) => {
    input.addEventListener('change', function () {
      const fileName = this.files[0]?.name || 'No file selected';
      const fileLabel = document.createElement('small');
      fileLabel.className = 'form-text text-muted mt-1';
      fileLabel.textContent = fileName;

      // Remove any existing label
      const existingLabel = this.parentElement.querySelector('small');
      if (existingLabel) {
        existingLabel.remove();
      }

      // Add the new label
      this.parentElement.appendChild(fileLabel);

      // Enable submit button if file is selected
      const submitBtn = this.closest('form')?.querySelector(
        'button[type="submit"]'
      );
      if (submitBtn && this.files.length > 0) {
        submitBtn.disabled = false;
      }
    });
  });
}

/**
 * Handles webcam stream errors and cleanup
 */
function initializeWebcamHandling() {
  const videoStream = document.querySelector('.video-container img');
  if (!videoStream) return;

  // Handle video stream errors
  videoStream.onerror = function () {
    videoStream.style.display = 'none';

    const errorMsg = document.createElement('div');
    errorMsg.className = 'alert alert-danger m-3';
    errorMsg.innerHTML = `
      <strong>Camera Error:</strong> Could not connect to webcam. 
      Please make sure your camera is connected and permissions are granted.
    `;

    videoStream.parentNode.appendChild(errorMsg);
  };

  // Handle resource cleanup on page leave
  window.addEventListener('beforeunload', () => {
    releaseCamera();
  });

  // Add handler for back button
  const backButton = document.getElementById('back-home-btn');
  if (backButton) {
    backButton.addEventListener('click', function (e) {
      e.preventDefault();
      releaseCamera().then(() => {
        window.location.href = '/';
      });
    });
  }
}

/**
 * Release camera resources
 * @returns {Promise} Resolves when camera is released
 */
function releaseCamera() {
  return fetch('/stop_stream', {
    method: 'GET',
    cache: 'no-cache',
  })
    .then((response) => {
      console.log('Camera resources released');
      return response;
    })
    .catch((error) => {
      console.error('Error releasing camera:', error);
    });
}

/**
 * Setup image upload form handling
 */
function initializeUploadForm() {
  const uploadForm = document.getElementById('upload-form');
  if (!uploadForm) return;

  const resultContainer = document.getElementById('result-container');
  const resultImage = document.getElementById('result-image');

  uploadForm.addEventListener('submit', async function (e) {
    e.preventDefault();

    const fileInput = document.getElementById('image-upload');
    const file = fileInput.files[0];

    if (!file) {
      showAlert('Please select an image to upload', 'warning');
      return;
    }

    // Validate file is an image
    if (!file.type.match('image.*')) {
      showAlert(
        'Please select a valid image file (JPEG, PNG, etc.)',
        'warning'
      );
      return;
    }

    // Validate file size (limit to 5MB)
    if (file.size > 5 * 1024 * 1024) {
      showAlert(
        'Image file is too large. Please select an image smaller than 5MB.',
        'warning'
      );
      return;
    }

    try {
      // Show loading indicator
      resultContainer.style.display = 'block';
      resultImage.style.display = 'none';

      // Create and show loading indicators
      const loadingElements = createLoadingElements();
      resultContainer
        .querySelector('.card-body')
        .appendChild(loadingElements.spinner);
      resultContainer
        .querySelector('.card-body')
        .appendChild(loadingElements.text);

      // Create FormData
      const formData = new FormData();
      formData.append('image', file);

      // Send API request
      console.log('Uploading image for detection...');
      const response = await fetch('/api/detect', {
        method: 'POST',
        body: formData,
      });

      // Handle errors
      if (!response.ok) {
        let errorMessage = `Error: ${response.status} ${response.statusText}`;

        try {
          const errorData = await response.json();
          errorMessage = errorData.error || errorMessage;
        } catch (jsonError) {
          // If JSON parsing fails, use the default message
        }

        throw new Error(errorMessage);
      }

      console.log('Image processed successfully');

      // Display result
      const blob = await response.blob();
      resultImage.src = URL.createObjectURL(blob);
      resultImage.alt = 'Detection Result';
      resultImage.style.display = 'block';

      // Remove loading elements
      document.getElementById('loading-spinner')?.remove();
      document.getElementById('loading-text')?.remove();
    } catch (error) {
      console.error('Error:', error);

      // Remove loading elements
      document.getElementById('loading-spinner')?.remove();
      document.getElementById('loading-text')?.remove();

      // Show error message
      showAlert(
        `Error processing image: ${error.message}`,
        'danger',
        resultContainer.querySelector('.card-body')
      );

      // Hide result image
      resultImage.src = '';
      resultImage.alt = '';
      resultImage.style.display = 'none';
    }
  });
}

/**
 * Creates loading spinner and text elements
 * @returns {Object} Object containing spinner and text elements
 */
function createLoadingElements() {
  // Create spinner
  const spinner = document.createElement('div');
  spinner.className = 'spinner';
  spinner.id = 'loading-spinner';

  // Create loading text
  const loadingText = document.createElement('div');
  loadingText.className = 'loading-text';
  loadingText.id = 'loading-text';
  loadingText.textContent = 'Processing image...';

  return { spinner, text: loadingText };
}

/**
 * Shows an alert message
 * @param {string} message - The message to display
 * @param {string} type - The alert type (success, danger, warning, info)
 * @param {HTMLElement} container - The container to append the alert to (optional)
 */
function showAlert(message, type = 'danger', container = null) {
  // Create alert element
  const alert = document.createElement('div');
  alert.className = `alert alert-${type} alert-dismissible fade show mt-3`;
  alert.innerHTML = `
    ${message}
    <button type="button" class="btn-close" data-bs-dismiss="alert" aria-label="Close"></button>
  `;

  // If container is provided, append to it
  if (container) {
    // Remove any existing alerts
    container.querySelectorAll('.alert').forEach((el) => el.remove());
    container.appendChild(alert);
  } else {
    // Otherwise, show at the top of the page
    const alertContainer = document.createElement('div');
    alertContainer.className = 'container mt-3';
    alertContainer.appendChild(alert);

    const mainContent = document.querySelector('main') || document.body;
    mainContent.insertBefore(alertContainer, mainContent.firstChild);
  }

  // Auto-dismiss after 5 seconds
  setTimeout(() => {
    alert.classList.remove('show');
    setTimeout(() => alert.remove(), 150);
  }, 5000);
}
