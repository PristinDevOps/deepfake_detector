// Wait for DOM to be fully loaded
document.addEventListener('DOMContentLoaded', function() {
    // Form elements
    const imageForm = document.getElementById('image-upload-form');
    const videoForm = document.getElementById('video-upload-form');
    
    // Results elements
    const loadingContainer = document.getElementById('loading-container');
    const resultsContainer = document.getElementById('results-container');
    const resultBadge = document.getElementById('result-badge');
    const confidenceBar = document.getElementById('confidence-bar');
    const resultImage = document.getElementById('result-image');
    const imageResult = document.getElementById('image-result');
    const videoResult = document.getElementById('video-result');
    const resultVideo = document.getElementById('result-video');
    
    // Video stats elements
    const videoStats = document.getElementById('video-stats');
    const totalFrames = document.getElementById('total-frames');
    const realFrames = document.getElementById('real-frames');
    const fakeFrames = document.getElementById('fake-frames');
    
    // Loading message
    const loadingMessage = document.getElementById('loading-message');
    
    // Handle image form submission
    imageForm.addEventListener('submit', function(e) {
        e.preventDefault();
        const formData = new FormData(this);
        
        // Show loading
        loadingContainer.classList.remove('d-none');
        resultsContainer.classList.add('d-none');
        loadingMessage.textContent = 'Analyzing image. This may take a few moments...';
        
        // Reset previous results
        resetResults();
        
        // Submit form data
        fetch('/upload', {
            method: 'POST',
            body: formData
        })
        .then(response => {
            if (!response.ok) {
                throw new Error(`HTTP error! Status: ${response.status}`);
            }
            return response.json();
        })
        .then(data => {
            // Hide loading
            loadingContainer.classList.add('d-none');
            
            if (data.error) {
                showError(data.error);
                return;
            }
            
            // Show results
            resultsContainer.classList.remove('d-none');
            
            // Show image result, hide video result
            imageResult.classList.remove('d-none');
            videoResult.classList.add('d-none');
            videoStats.classList.add('d-none');
            
            // Set result image
            resultImage.src = data.result_url;
            
            // Display result badge
            displayResultBadge(data.is_real, data.confidence);
        })
        .catch(error => {
            loadingContainer.classList.add('d-none');
            showError('Error processing image: ' + error.message);
            console.error('Error:', error);
        });
    });
    
    // Handle video form submission
    videoForm.addEventListener('submit', function(e) {
        e.preventDefault();
        const formData = new FormData(this);
        
        // Show loading
        loadingContainer.classList.remove('d-none');
        resultsContainer.classList.add('d-none');
        loadingMessage.textContent = 'Processing video. This may take several minutes depending on the video length...';
        
        // Reset previous results
        resetResults();
        
        // Submit form data
        fetch('/upload', {
            method: 'POST',
            body: formData
        })
        .then(response => {
            if (!response.ok) {
                throw new Error(`HTTP error! Status: ${response.status}`);
            }
            return response.json();
        })
        .then(data => {
            // Hide loading
            loadingContainer.classList.add('d-none');
            
            if (data.error) {
                showError(data.error);
                return;
            }
            
            // Show results
            resultsContainer.classList.remove('d-none');
            
            // Show video result, hide image result
            imageResult.classList.add('d-none');
            videoResult.classList.remove('d-none');
            videoStats.classList.remove('d-none');
            
            // Set result video
            resultVideo.src = data.result_url;
            resultVideo.load();
            
            // Set video stats
            totalFrames.textContent = data.total_frames || 0;
            realFrames.textContent = data.real_frames || 0;
            fakeFrames.textContent = data.fake_frames || 0;
            
            // Display result badge
            displayResultBadge(data.is_real, data.confidence);
        })
        .catch(error => {
            loadingContainer.classList.add('d-none');
            showError('Error processing video: ' + error.message);
            console.error('Error:', error);
        });
    });
    
    // Function to display result badge and confidence
    function displayResultBadge(isReal, confidence) {
        // Set badge text and class
        resultBadge.textContent = isReal ? 'REAL' : 'FAKE';
        resultBadge.className = 'badge rounded-pill p-3 mb-3 fs-5';
        resultBadge.classList.add(isReal ? 'real' : 'fake');
        resultBadge.classList.add(isReal ? 'bg-success' : 'bg-danger');
        
        // Set confidence bar
        confidenceBar.style.width = confidence + '%';
        confidenceBar.setAttribute('aria-valuenow', confidence);
        confidenceBar.textContent = confidence.toFixed(1) + '%';
        confidenceBar.className = 'progress-bar';
        confidenceBar.classList.add(isReal ? 'real' : 'fake');
        confidenceBar.classList.add(isReal ? 'bg-success' : 'bg-danger');
    }
    
    // Function to reset results
    function resetResults() {
        resultImage.src = '';
        resultVideo.src = '';
        resultBadge.textContent = '';
        confidenceBar.style.width = '0%';
        confidenceBar.setAttribute('aria-valuenow', 0);
        confidenceBar.textContent = '0%';
        totalFrames.textContent = '0';
        realFrames.textContent = '0';
        fakeFrames.textContent = '0';
    }
    
    // Function to show error message
    function showError(message) {
        const errorElement = document.createElement('div');
        errorElement.className = 'alert alert-danger alert-dismissible fade show';
        errorElement.innerHTML = `
            <strong>Error:</strong> ${message}
            <button type="button" class="btn-close" data-bs-dismiss="alert" aria-label="Close"></button>
        `;
        
        // Insert after the form card
        const formCard = document.querySelector('.card');
        formCard.parentNode.insertBefore(errorElement, formCard.nextSibling);
        
        // Auto-dismiss after 8 seconds
        setTimeout(() => {
            errorElement.classList.remove('show');
            setTimeout(() => errorElement.remove(), 500);
        }, 8000);
    }
}); 