/**
 * OpenBackground - Frontend Application
 * Vanilla JavaScript dashboard for background removal service
 */

class OpenBackground {
    constructor() {
        this.apiKey = localStorage.getItem('openbackground_api_key') || '';
        this.selectedFile = null;
        this.processedImageBlob = null;
        
        this.elements = {
            apiKeyInput: document.getElementById('apiKey'),
            toggleApiKey: document.getElementById('toggleApiKey'),
            modelSelect: document.getElementById('modelSelect'),
            deviceBadge: document.getElementById('deviceBadge'),
            uploadArea: document.getElementById('uploadArea'),
            fileInput: document.getElementById('fileInput'),
            returnMask: document.getElementById('returnMask'),
            processBtn: document.getElementById('processBtn'),
            originalImage: document.getElementById('originalImage'),
            processedImage: document.getElementById('processedImage'),
            originalWrapper: document.getElementById('originalWrapper'),
            processedWrapper: document.getElementById('processedWrapper'),
            downloadBtn: document.getElementById('downloadBtn'),
            processingTime: document.getElementById('processingTime'),
            refreshStats: document.getElementById('refreshStats'),
            totalRequests: document.getElementById('totalRequests'),
            successRate: document.getElementById('successRate'),
            avgTime: document.getElementById('avgTime'),
            uptime: document.getElementById('uptime'),
            modelBars: document.getElementById('modelBars'),
            toastContainer: document.getElementById('toastContainer'),
        };
        
        this.init();
    }
    
    init() {
        this.setupEventListeners();
        this.loadApiKey();
        this.fetchModels();
        this.fetchStats();
        
        // Auto-refresh stats every 30 seconds
        setInterval(() => this.fetchStats(), 30000);
    }
    
    setupEventListeners() {
        // API Key
        this.elements.apiKeyInput.addEventListener('input', (e) => {
            this.apiKey = e.target.value;
            localStorage.setItem('openbackground_api_key', this.apiKey);
        });
        
        this.elements.toggleApiKey.addEventListener('click', () => {
            const input = this.elements.apiKeyInput;
            input.type = input.type === 'password' ? 'text' : 'password';
        });
        
        // File Upload
        this.elements.uploadArea.addEventListener('click', () => {
            this.elements.fileInput.click();
        });
        
        this.elements.fileInput.addEventListener('change', (e) => {
            if (e.target.files.length > 0) {
                this.handleFileSelect(e.target.files[0]);
            }
        });
        
        // Drag and Drop
        this.elements.uploadArea.addEventListener('dragover', (e) => {
            e.preventDefault();
            this.elements.uploadArea.classList.add('dragover');
        });
        
        this.elements.uploadArea.addEventListener('dragleave', () => {
            this.elements.uploadArea.classList.remove('dragover');
        });
        
        this.elements.uploadArea.addEventListener('drop', (e) => {
            e.preventDefault();
            this.elements.uploadArea.classList.remove('dragover');
            
            if (e.dataTransfer.files.length > 0) {
                this.handleFileSelect(e.dataTransfer.files[0]);
            }
        });
        
        // Process Button
        this.elements.processBtn.addEventListener('click', () => {
            this.processImage();
        });
        
        // Download Button
        this.elements.downloadBtn.addEventListener('click', () => {
            this.downloadImage();
        });
        
        // Model Select
        this.elements.modelSelect.addEventListener('change', (e) => {
            if (e.target.value) {
                this.loadModel(e.target.value);
            }
        });
        
        // Refresh Stats
        this.elements.refreshStats.addEventListener('click', () => {
            this.fetchStats();
        });
    }
    
    loadApiKey() {
        if (this.apiKey) {
            this.elements.apiKeyInput.value = this.apiKey;
        }
    }
    
    getHeaders() {
        return {
            'X-API-Key': this.apiKey,
        };
    }
    
    async fetchModels() {
        try {
            const response = await fetch('/api/v1/models', {
                headers: this.getHeaders(),
            });
            
            if (!response.ok) {
                throw new Error('Failed to fetch models');
            }
            
            const data = await response.json();
            this.updateModelSelector(data);
            this.updateDeviceBadge(data.device, data.cuda_available);
        } catch (error) {
            console.error('Error fetching models:', error);
            this.showToast('Failed to fetch models. Check your API key.', 'error');
        }
    }
    
    updateModelSelector(data) {
        const select = this.elements.modelSelect;
        select.innerHTML = '';
        
        data.available_models.forEach(model => {
            const option = document.createElement('option');
            option.value = model.name;
            option.textContent = model.name.split('/').pop();
            option.selected = model.is_current;
            
            if (model.loaded) {
                option.textContent += ' (loaded)';
            }
            
            select.appendChild(option);
        });
    }
    
    updateDeviceBadge(device, cudaAvailable) {
        const badge = this.elements.deviceBadge;
        badge.textContent = device.toUpperCase();
        
        if (device === 'cuda') {
            badge.classList.add('gpu');
        } else {
            badge.classList.remove('gpu');
        }
    }
    
    async loadModel(modelName) {
        try {
            this.showToast(`Loading model: ${modelName}...`, 'info');
            
            const response = await fetch(`/api/v1/models/load`, {
                method: 'POST',
                headers: {
                    ...this.getHeaders(),
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({ model_name: modelName }),
            });
            
            if (!response.ok) {
                throw new Error('Failed to load model');
            }
            
            const data = await response.json();
            this.showToast(data.message, 'success');
            this.fetchModels();
        } catch (error) {
            console.error('Error loading model:', error);
            this.showToast('Failed to load model', 'error');
        }
    }
    
    handleFileSelect(file) {
        if (!file.type.startsWith('image/')) {
            this.showToast('Please select an image file', 'error');
            return;
        }
        
        this.selectedFile = file;
        this.elements.processBtn.disabled = false;
        
        // Show preview
        const reader = new FileReader();
        reader.onload = (e) => {
            this.elements.originalImage.src = e.target.result;
            this.elements.originalImage.classList.add('visible');
            this.elements.originalWrapper.classList.add('has-image');
        };
        reader.readAsDataURL(file);
        
        // Clear processed image
        this.elements.processedImage.classList.remove('visible');
        this.elements.processedWrapper.classList.remove('has-image');
        this.elements.downloadBtn.disabled = true;
        this.elements.processingTime.textContent = '';
        this.processedImageBlob = null;
    }
    
    async processImage() {
        if (!this.selectedFile) {
            this.showToast('Please select an image first', 'error');
            return;
        }
        
        if (!this.apiKey) {
            this.showToast('Please enter your API key', 'error');
            return;
        }
        
        const btn = this.elements.processBtn;
        btn.classList.add('loading');
        btn.disabled = true;
        
        try {
            const formData = new FormData();
            formData.append('file', this.selectedFile);
            formData.append('model', this.elements.modelSelect.value);
            formData.append('return_mask', this.elements.returnMask.checked);
            
            const response = await fetch('/api/v1/remove-background', {
                method: 'POST',
                headers: this.getHeaders(),
                body: formData,
            });
            
            if (!response.ok) {
                const error = await response.json();
                throw new Error(error.detail || 'Processing failed');
            }
            
            // Get metadata from headers
            const processingTime = response.headers.get('X-Processing-Time-Ms');
            const modelUsed = response.headers.get('X-Model-Used');
            
            // Get image blob
            this.processedImageBlob = await response.blob();
            const imageUrl = URL.createObjectURL(this.processedImageBlob);
            
            // Update UI
            this.elements.processedImage.src = imageUrl;
            this.elements.processedImage.classList.add('visible');
            this.elements.processedWrapper.classList.add('has-image');
            this.elements.downloadBtn.disabled = false;
            
            if (processingTime) {
                this.elements.processingTime.textContent = `${parseFloat(processingTime).toFixed(0)}ms`;
            }
            
            this.showToast('Image processed successfully!', 'success');
            this.fetchStats();
            
        } catch (error) {
            console.error('Error processing image:', error);
            this.showToast(error.message || 'Failed to process image', 'error');
        } finally {
            btn.classList.remove('loading');
            btn.disabled = false;
        }
    }
    
    downloadImage() {
        if (!this.processedImageBlob) {
            return;
        }
        
        const url = URL.createObjectURL(this.processedImageBlob);
        const a = document.createElement('a');
        a.href = url;
        a.download = `processed_${Date.now()}.png`;
        document.body.appendChild(a);
        a.click();
        document.body.removeChild(a);
        URL.revokeObjectURL(url);
    }
    
    async fetchStats() {
        try {
            const response = await fetch('/api/v1/stats', {
                headers: this.getHeaders(),
            });
            
            if (!response.ok) {
                throw new Error('Failed to fetch stats');
            }
            
            const data = await response.json();
            this.updateStats(data);
        } catch (error) {
            console.error('Error fetching stats:', error);
        }
    }
    
    updateStats(data) {
        this.elements.totalRequests.textContent = data.total_requests.toLocaleString();
        this.elements.successRate.textContent = `${data.success_rate_percent}%`;
        this.elements.avgTime.textContent = `${Math.round(data.average_processing_time_ms)}ms`;
        this.elements.uptime.textContent = this.formatUptime(data.uptime_seconds);
        
        this.updateModelBars(data.requests_per_model);
    }
    
    formatUptime(seconds) {
        if (seconds < 60) {
            return `${Math.round(seconds)}s`;
        } else if (seconds < 3600) {
            return `${Math.round(seconds / 60)}m`;
        } else if (seconds < 86400) {
            return `${Math.round(seconds / 3600)}h`;
        } else {
            return `${Math.round(seconds / 86400)}d`;
        }
    }
    
    updateModelBars(requestsPerModel) {
        const container = this.elements.modelBars;
        
        if (!requestsPerModel || Object.keys(requestsPerModel).length === 0) {
            container.innerHTML = '<p class="no-data">No data yet</p>';
            return;
        }
        
        const maxCount = Math.max(...Object.values(requestsPerModel));
        
        container.innerHTML = Object.entries(requestsPerModel)
            .sort((a, b) => b[1] - a[1])
            .map(([model, count]) => {
                const percentage = maxCount > 0 ? (count / maxCount) * 100 : 0;
                const shortName = model.split('/').pop();
                
                return `
                    <div class="model-bar">
                        <span class="model-bar-label" title="${model}">${shortName}</span>
                        <div class="model-bar-track">
                            <div class="model-bar-fill" style="width: ${percentage}%"></div>
                        </div>
                        <span class="model-bar-count">${count}</span>
                    </div>
                `;
            })
            .join('');
    }
    
    showToast(message, type = 'info') {
        const toast = document.createElement('div');
        toast.className = `toast ${type}`;
        toast.innerHTML = `
            <span class="toast-message">${message}</span>
            <button class="toast-close">&times;</button>
        `;
        
        const closeBtn = toast.querySelector('.toast-close');
        closeBtn.addEventListener('click', () => {
            toast.remove();
        });
        
        this.elements.toastContainer.appendChild(toast);
        
        // Auto-remove after 5 seconds
        setTimeout(() => {
            if (toast.parentNode) {
                toast.remove();
            }
        }, 5000);
    }
}

// Initialize app when DOM is ready
document.addEventListener('DOMContentLoaded', () => {
    window.app = new OpenBackground();
});

