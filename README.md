# 🌊 Invisible Watermarking Tool

A modern, web-based application for embedding and extracting invisible watermarks from images using Discrete Cosine Transform (DCT) technology. Built with React frontend and Flask backend.

![Invisible Watermarking Tool](https://img.shields.io/badge/React-18+-blue) ![Flask](https://img.shields.io/badge/Flask-3.0+-green) ![Python](https://img.shields.io/badge/Python-3.8+-yellow)

## ✨ Features

### 🎨 **Modern UI/UX**
- **3D Animated Background**: Beautiful Prism component with WebGL animations
- **Glass Morphism Design**: Semi-transparent elements with backdrop blur effects
- **Animated Buttons**: Custom TrialButton component with gradient animations
- **Responsive Design**: Works perfectly on desktop, tablet, and mobile devices
- **Dark Theme**: Professional dark interface with blue accents

### 🔧 **Watermarking Technology**
- **DCT Algorithm**: Uses Discrete Cosine Transform for frequency domain embedding
- **Invisible Watermarks**: Embeds watermarks that are imperceptible to the human eye
- **Robust Extraction**: Reliable watermark recovery from processed images
- **Multiple Formats**: Supports PNG, JPG, and JPEG image formats

### 🚀 **Technical Features**
- **Real-time Processing**: Fast watermark embedding and extraction
- **Drag & Drop Upload**: Intuitive file upload with visual feedback
- **Progress Indicators**: Loading states and progress feedback
- **Error Handling**: Comprehensive error messages and validation
- **Cross-browser Support**: Works on all modern browsers

## 🏗️ Project Structure

```
invisible-watermarking/
├── watermark-app/                 # React Frontend
│   ├── src/
│   │   ├── components/
│   │   │   ├── EmbedWatermark.jsx    # Watermark embedding interface
│   │   │   ├── ExtractWatermark.jsx  # Watermark extraction interface
│   │   │   ├── FileUpload.jsx        # Drag & drop file upload
│   │   │   ├── Prism.jsx            # 3D animated background
│   │   │   └── TrialButton.jsx      # Animated button component
│   │   ├── App.jsx                   # Main application component
│   │   ├── App.css                   # Global styles and animations
│   │   └── main.jsx                  # React entry point
│   ├── public/
│   │   └── index.html               # HTML template
│   ├── package.json                 # Dependencies and scripts
│   └── vite.config.js              # Vite configuration
├── backend/                        # Flask Backend
│   ├── app.py                      # Main Flask application
│   └── requirements.txt            # Python dependencies
└── README.md                       # This file
```

## 🚀 Quick Start

### Prerequisites
- **Node.js** (v16 or higher)
- **Python** (v3.8 or higher)
- **npm** or **yarn**

### 🔧 Installation

#### 1. Clone the Repository
```bash
git clone https://github.com/yourusername/invisible-watermarking.git
cd invisible-watermarking
```

#### 2. Setup Backend (Flask API)
```bash
# Navigate to backend directory
cd backend

# Create virtual environment (recommended)
python -m venv venv

# Activate virtual environment
# Windows:
venv\Scripts\activate
# macOS/Linux:
source venv/bin/activate

# Install Python dependencies
pip install -r requirements.txt

# Start Flask server
python app.py
```
The backend server will start on `http://localhost:5000`

#### 3. Setup Frontend (React App)
```bash
# Open new terminal and navigate to frontend
cd watermark-app

# Install dependencies
npm install

# Start development server
npm run dev
```
The frontend will start on `http://localhost:5173`

### 🎯 Usage

1. **Open Application**: Navigate to `http://localhost:5173` in your browser
2. **Choose Operation**: 
   - Click "Embed Watermark" to add watermarks to images
   - Click "Extract Watermark" to recover watermarks from images
3. **Upload Files**: 
   - Drag and drop images onto the upload area
   - Or click to browse and select files
4. **Process**: Click the animated "Process" button to start watermarking
5. **Download**: Results will be automatically downloaded

## 🧩 Component Documentation

### 🎨 **App.jsx** - Main Application Component
**Purpose**: Root component managing navigation, 3D background, and overall layout

**Key Features**:
- **Tab-based Navigation**: Switch between embed/extract modes
- **Custom SVG Logo**: Animated logo with walking figure and water ripples
- **Responsive Layout**: Side-by-side desktop layout, stacked mobile layout
- **3D Background Integration**: Manages Prism component rendering

**State Management**:
```javascript
const [activeTab, setActiveTab] = useState('embed'); // Navigation state
```

**Navigation System**:
- Handles tab switching between watermark operations
- Applies active styles and smooth transitions
- Responsive button layout with proper spacing

### 🖼️ **EmbedWatermark.jsx** - Watermark Embedding Interface
**Purpose**: Handles watermark embedding workflow with file upload and API communication

**Key Features**:
- **File Validation**: Ensures proper image formats (PNG, JPG, JPEG)
- **API Communication**: Sends FormData to Flask backend via axios
- **Loading States**: Visual feedback during processing
- **Error Handling**: Comprehensive error messages and recovery

**API Integration**:
```javascript
// FormData preparation for multipart/form-data upload
const formData = new FormData();
formData.append('image', file);
formData.append('watermark', watermarkText);

// Axios POST request with proper headers
const response = await axios.post('http://localhost:5000/embed', formData, {
  headers: { 'Content-Type': 'multipart/form-data' },
  responseType: 'blob' // Handle binary image response
});
```

**State Management**:
- `watermarkText`: User input for watermark content
- `file`: Selected image file for processing
- `loading`: Loading state for UI feedback

### 🎁 **TrialButton.jsx** - Animated Button Component
**Purpose**: Reusable button with advanced CSS animations and gradient effects

**Key Features**:
- **CSS-in-JS Approach**: Embedded styles with dynamic properties
- **Gradient Animations**: Rotating conic gradients with shimmer effects
- **Hover Effects**: Scale transforms and color transitions
- **Accessibility**: Proper focus states and disabled handling

**Animation System**:
```css
/* Conic gradient rotation for shimmer effect */
@keyframes rotate {
  0% { --angle: 0deg; }
  100% { --angle: 360deg; }
}

/* CSS custom properties for dynamic styling */
.trial-button {
  --angle: 0deg;
  background: conic-gradient(from var(--angle), /* gradient stops */);
}
```

**Performance Optimizations**:
- GPU-accelerated transforms
- Efficient CSS custom properties
- Minimal reflows and repaints

### 📁 **FileUpload.jsx** - Drag & Drop Upload Component
**Purpose**: Intuitive file upload interface with drag-and-drop functionality

**Key Features**:
- **Drag & Drop**: Visual feedback for drag states
- **File Validation**: Format and size checking
- **Preview**: Image preview before processing
- **Error Handling**: Clear validation messages

### 🌈 **Prism.jsx** - 3D Animated Background
**Purpose**: WebGL-based 3D background using OGL library

**Key Features**:
- **WebGL Rendering**: Hardware-accelerated 3D graphics
- **Rotation Animation**: Smooth prism rotation
- **Responsive Canvas**: Adapts to container size
- **Performance Optimized**: Efficient rendering loop

## 🔌 API Documentation

### Backend Endpoints

#### **POST** `/embed`
Embeds watermark into image using DCT algorithm

**Request**:
- **Content-Type**: `multipart/form-data`
- **Body**:
  - `image`: Image file (PNG/JPG/JPEG)
  - `watermark`: Text string to embed

**Response**:
- **Content-Type**: `image/png`
- **Body**: Watermarked image binary data

**Example**:
```javascript
const formData = new FormData();
formData.append('image', imageFile);
formData.append('watermark', 'Secret Message');

const response = await fetch('http://localhost:5000/embed', {
  method: 'POST',
  body: formData
});
```

#### **POST** `/extract`
Extracts watermark from watermarked image

**Request**:
- **Content-Type**: `multipart/form-data`
- **Body**:
  - `image`: Watermarked image file

**Response**:
- **Content-Type**: `application/json`
- **Body**: `{"watermark": "extracted_text"}`

**Example**:
```javascript
const formData = new FormData();
formData.append('image', watermarkedImage);

const response = await fetch('http://localhost:5000/extract', {
  method: 'POST',
  body: formData
});
const result = await response.json();
console.log(result.watermark); // Extracted watermark text
```

### Error Handling

**Common Error Responses**:
- `400 Bad Request`: Invalid file format or missing data
- `500 Internal Server Error`: Processing errors

**Frontend Error Handling**:
```javascript
try {
  const response = await axios.post('/embed', formData);
  // Handle success
} catch (error) {
  if (error.response) {
    // Server responded with error status
    setError(`Server error: ${error.response.status}`);
  } else if (error.request) {
    // Network error
    setError('Network error. Please check your connection.');
  } else {
    // Other error
    setError('An unexpected error occurred.');
  }
}
```

## 🛠️ Technical Architecture

### Watermarking Algorithm - DCT Implementation

The application implements **Discrete Cosine Transform (DCT)** watermarking for robust, invisible watermark embedding:

#### **Embedding Process**:
1. **Color Space Conversion**: RGB → YCrCb for frequency domain processing
2. **DCT Application**: 8x8 block-wise DCT on luminance channel
3. **Frequency Modification**: Mid-frequency coefficient adjustment
4. **Reconstruction**: Inverse DCT and color space conversion back to RGB

#### **Technical Parameters**:
- **Scaling Factor**: α = 0.05 for optimal invisibility vs. robustness
- **Block Size**: 8x8 pixels for standard DCT processing
- **Channel**: Y (luminance) channel for perceptual invisibility

```python
# Core DCT Watermarking Implementation
def embed_watermark_dct(image, watermark_text):
    # Convert to YCrCb color space
    ycrcb_img = cv2.cvtColor(image, cv2.COLOR_BGR2YCrCb)
    y_channel = ycrcb_img[:, :, 0].astype(np.float32)
    
    # Apply DCT to luminance channel
    dct_img = cv2.dct(y_channel)
    
    # Embed watermark in DCT coefficients
    # Implementation details in backend/app.py
```

### Frontend Architecture

#### **React Component Hierarchy**:
```
App.jsx (Root)
├── Prism.jsx (3D Background)
├── Navigation (Tab System)
├── EmbedWatermark.jsx
│   ├── FileUpload.jsx
│   └── TrialButton.jsx
└── ExtractWatermark.jsx
    ├── FileUpload.jsx
    └── TrialButton.jsx
```

#### **State Management Pattern**:
- **Local State**: Component-specific UI state (loading, errors)
- **Props Drilling**: Parent-child component communication
- **Event Handling**: User interaction and file processing

#### **Styling Approach**:
- **CSS-in-JS**: Embedded styles in TrialButton component
- **Global CSS**: App.css for layout and animations
- **Glass Morphism**: backdrop-filter for modern UI effects
- **Responsive Design**: CSS Grid and Flexbox for adaptive layouts

### Backend Architecture

#### **Flask API Structure**:
```python
# app.py - Main Flask Application
from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
import cv2
import numpy as np

app = Flask(__name__)
CORS(app)  # Enable Cross-Origin Resource Sharing

@app.route('/embed', methods=['POST'])
def embed_watermark():
    # DCT watermark embedding logic
    
@app.route('/extract', methods=['POST']) 
def extract_watermark():
    # DCT watermark extraction logic
```

#### **Image Processing Pipeline**:
1. **File Reception**: Multipart form data handling
2. **Format Validation**: PNG/JPG/JPEG support
3. **DCT Processing**: Frequency domain watermarking
4. **Response Generation**: Binary image or JSON data

### Performance Optimizations

#### **Frontend**:
- **Vite Build Tool**: Fast development and optimized production builds
- **Component Memoization**: React optimization patterns
- **Lazy Loading**: Dynamic component imports
- **Asset Optimization**: Image compression and bundling

#### **Backend**:
- **OpenCV Optimizations**: Efficient image processing algorithms
- **Memory Management**: Proper array handling with NumPy
- **Error Handling**: Graceful failure recovery
- **File Cleanup**: Temporary file management

#### **3D Graphics (Prism)**:
- **WebGL Rendering**: Hardware-accelerated graphics
- **RAF Animation**: RequestAnimationFrame for smooth animations
- **Canvas Optimization**: Efficient rendering loop
- **Memory Management**: Proper WebGL resource cleanup

## 🔧 Development Tools & Dependencies

### Frontend Dependencies
```json
{
  "react": "^18.2.0",
  "vite": "^5.0.0", 
  "axios": "^1.6.0",
  "ogl": "^1.0.0"
}
```

### Backend Dependencies
```txt
Flask==3.0.0
flask-cors==4.0.0
opencv-python==4.8.1.78
numpy==1.24.3
Pillow==10.1.0
```

### Development Environment
- **Node.js**: v16+ for modern JavaScript features
- **Python**: v3.8+ for OpenCV compatibility
- **VS Code**: Recommended IDE with extensions
- **Git**: Version control with proper gitignore

## 🚀 Production Deployment

### Frontend Build Process
```bash
# Build optimized production bundle
npm run build

# Serves static files from dist/
npm run preview
```

### Backend Production Setup
```bash
# Install production WSGI server
pip install gunicorn

# Run with Gunicorn
gunicorn -w 4 -b 0.0.0.0:5000 app:app
```

### Environment Configuration
```bash
# Environment variables for production
export FLASK_ENV=production
export CORS_ORIGINS="https://yourdomain.com"
export MAX_CONTENT_LENGTH=16777216  # 16MB
```

## 🔍 Security Considerations

### File Upload Security
- **Size Limits**: 16MB maximum file size
- **Format Validation**: Strict image format checking
- **File Type Detection**: MIME type validation
- **Temporary Files**: Automatic cleanup after processing

### API Security
- **CORS Configuration**: Proper origin restrictions
- **Input Validation**: File and parameter validation
- **Error Handling**: No sensitive information exposure
- **Rate Limiting**: Consider implementing for production

### Frontend Security
- **XSS Prevention**: Proper data sanitization
- **CSRF Protection**: Token-based protection (if needed)
- **Secure Headers**: Content Security Policy implementation
- **HTTPS Enforcement**: SSL/TLS for production

## 🐛 Troubleshooting Guide

### Common Frontend Issues

#### **Build Errors**:
```bash
# Clear node modules and reinstall
rm -rf node_modules package-lock.json
npm install

# Clear Vite cache
npm run dev -- --force
```

#### **API Connection Issues**:
- Verify backend server is running on port 5000
- Check browser console for CORS errors
- Ensure proper Content-Type headers

### Common Backend Issues

#### **OpenCV Installation**:
```bash
# If OpenCV installation fails
pip uninstall opencv-python
pip install opencv-python-headless

# For compatibility issues
pip install numpy==1.24.3
```

#### **Flask CORS Issues**:
```python
# Explicit CORS configuration
from flask_cors import CORS, cross_origin

app = Flask(__name__)
CORS(app, origins=['http://localhost:5173'])
```

### Performance Issues

#### **Large File Processing**:
- Implement file size validation
- Add progress indicators for large files
- Consider implementing chunked uploads

#### **Memory Usage**:
- Monitor memory usage with large images
- Implement proper garbage collection
- Consider image resizing for very large files

## 📚 Additional Resources

### Learning Resources
- **React Documentation**: https://react.dev/
- **Flask Documentation**: https://flask.palletsprojects.com/
- **OpenCV Python Tutorials**: https://docs.opencv.org/
- **DCT Watermarking Papers**: Academic research on frequency domain watermarking

### Extension Ideas
- **Batch Processing**: Multiple file watermarking
- **Advanced Algorithms**: DWT, SVD-based watermarking  
- **Quality Metrics**: PSNR, SSIM evaluation
- **User Authentication**: Login system for watermark tracking
- **Cloud Storage**: Integration with cloud services

## 📄 License & Credits

This project is developed for educational and research purposes. 

**Technologies Used**:
- React.js & Vite for modern frontend development
- Flask for lightweight Python web API
- OpenCV for computer vision and image processing
- OGL for WebGL-based 3D graphics
- CSS3 for modern UI animations and effects

**Academic References**:
- DCT-based watermarking research papers
- Digital image processing fundamentals
- Web security best practices

---

**Made with ❤️ using React, Flask, and OpenCV**
