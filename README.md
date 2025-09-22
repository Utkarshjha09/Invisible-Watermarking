# 🌊 Invisible Watermarking Tool

A modern, web-based application for embedding and extracting invisible watermarks from images using advanced DCT (Discrete Cosine #### 3.4 Install Dependencies
```bash
# Install npm packages
npm install
```

#### 3.5 Start Development Server
```bash
npm run dev
```
✅ **Frontend will start on `http://localhost:5173`**

### Step 4: Access the Application
Open your web browser and navigate to `http://localhost:5173`

## 🔧 Environment Configuration

### **Backend Environment Variables** (`backend/.env`)

| Variable | Description | Default | Example |
|----------|-------------|---------|---------|
| `FLASK_ENV` | Flask environment mode | `development` | `production` |
| `FLASK_DEBUG` | Enable/disable debug mode | `true` | `false` |
| `HOST` | Server host address | `localhost` | `0.0.0.0` |
| `PORT` | Server port number | `5000` | `8080` |
| `CORS_ORIGINS` | Allowed CORS origins | `http://localhost:5173` | `https://yourdomain.com` |
| `MAX_CONTENT_LENGTH` | Max file upload size (bytes) | `16777216` | `33554432` |
| `SECRET_KEY` | Flask secret key | `your-secret-key` | `random-secure-key` |
| `FREQ_ALPHA` | Frequency domain embedding strength | `0.08` | `0.10` |
| `BLOCK_DCT_ALPHA` | Block DCT embedding strength | `0.25` | `0.30` |
| `DEFAULT_DELTA` | Quantized method default delta | `0.05` | `0.06` |
| `MAX_WATERMARK_SIZE` | Maximum watermark dimensions | `64` | `128` |
| `LOG_LEVEL` | Logging verbosity | `DEBUG` | `INFO` |

### **Frontend Environment Variables** (`watermark-app/.env`)

| Variable | Description | Default | Example |
|----------|-------------|---------|---------|
| `VITE_API_URL` | Backend API base URL | `http://localhost:5000` | `https://api.yourdomain.com` |
| `VITE_API_TIMEOUT` | API request timeout (ms) | `30000` | `60000` |
| `VITE_MAX_FILE_SIZE` | Max file upload size (bytes) | `16777216` | `33554432` |
| `VITE_DEFAULT_ALGORITHM` | Default watermarking algorithm | `block_dct_q` | `freq` |
| `VITE_DEFAULT_DELTA` | Default delta value | `0.05` | `0.04` |
| `VITE_ENABLE_3D_BACKGROUND` | Enable 3D background | `true` | `false` |
| `VITE_ENABLE_DEBUG_MODE` | Enable debug features | `true` | `false` |
| `VITE_APP_TITLE` | Application title | `Invisible Watermarking Tool` | `Your App Name` |

### **Environment Setup for Different Stages**

#### **Development Environment:**
```bash
# Backend
FLASK_ENV=development
FLASK_DEBUG=true
LOG_LEVEL=DEBUG

# Frontend  
VITE_ENABLE_DEBUG_MODE=true
VITE_DEV_TOOLS=true
```

#### **Production Environment:**
```bash
# Backend
FLASK_ENV=production
FLASK_DEBUG=false
LOG_LEVEL=INFO
SECRET_KEY=your-super-secure-production-key
CORS_ORIGINS=https://yourdomain.com

# Frontend
VITE_API_URL=https://your-production-api.com
VITE_ENABLE_DEBUG_MODE=false
VITE_DEV_TOOLS=false
```orithms. Built with React frontend and Flask backend, featuring multiple watermarking algorithms for optimal security and robustness.

![Invisible Watermarking Tool](https://img.shields.io/badge/React-19+-blue) ![Flask](https://img.shields.io/badge/Flask-2.3+-green) ![Python](https://img.shields.io/badge/Python-3.8+-yellow) ![OpenCV](https://img.shields.io/badge/OpenCV-4.10+-orange)

## ✨ Features

### 🎨 **Modern UI/UX**
- **3D Animated Background**: Beautiful Prism component with WebGL animations
- **Glass Morphism Design**: Semi-transparent elements with backdrop blur effects
- **Animated Buttons**: Custom TrialButton component with gradient animations
- **Responsive Design**: Works perfectly on desktop, tablet, and mobile devices
- **Dark Theme**: Professional dark interface with blue accents
- **Algorithm Selection**: Choose from multiple watermarking algorithms
- **Strength Control**: Adjustable embedding strength with delta parameter

### 🔧 **Advanced Watermarking Technology**
- **Multiple Algorithms**: 
  - **Quantized Block DCT** (Recommended): Bit-level embedding for maximum accuracy
  - **Block DCT**: Amplitude-based embedding for standard use cases
  - **Frequency Domain**: FFT-based embedding for specialized applications
- **Dual Watermark Support**: Both text and image watermarks
- **Auto-Detection**: Automatic watermark type detection during extraction
- **Invisible Embedding**: Watermarks imperceptible to human eye
- **Robust Extraction**: Reliable recovery with multi-tier fallback system
- **Multiple Formats**: Supports PNG, JPG, and JPEG image formats

### 🚀 **Technical Features**
- **Real-time Processing**: Fast watermark embedding and extraction
- **Drag & Drop Upload**: Intuitive file upload with visual feedback
- **Progress Indicators**: Loading states and progress feedback
- **Error Handling**: Comprehensive error messages and validation
- **Cross-browser Support**: Works on all modern browsers
- **Debug Logging**: Detailed extraction metrics for troubleshooting

## 🏗️ Project Structure

```
invisible-watermarking/
├── backend/                        # Flask Backend
│   ├── app.py                      # Main Flask application with watermarking APIs
│   ├── requirements.txt            # Python dependencies
│   └── test_watermarking.py        # Backend testing utilities
├── watermark-app/                  # React Frontend
│   ├── src/
│   │   ├── components/
│   │   │   ├── EmbedWatermark.jsx    # Watermark embedding interface
│   │   │   ├── ExtractWatermark.jsx  # Watermark extraction interface
│   │   │   ├── FileUpload.jsx        # Drag & drop file upload
│   │   │   ├── Prism.jsx            # 3D animated background
│   │   │   ├── SimpleBackground.jsx  # Alternative background
│   │   │   └── TrialButton.jsx      # Animated button component
│   │   ├── App.jsx                   # Main application component
│   │   ├── App.css                   # Global styles and animations
│   │   └── main.jsx                  # React entry point
│   ├── public/
│   │   ├── index.html               # HTML template
│   │   └── images/                  # Logo and assets
│   ├── package.json                 # Dependencies and scripts
│   └── vite.config.js              # Vite configuration
├── FinalProject.py                 # Original desktop version (legacy)
├── .venv/                         # Python virtual environment
└── README.md                      # This documentation
```

## 🚀 Quick Start

### Prerequisites
- **Node.js** (v16 or higher) - [Download here](https://nodejs.org/)
- **Python** (v3.8 or higher) - [Download here](https://python.org/)
- **Git** - [Download here](https://git-scm.com/)

### 📋 System Requirements
- **RAM**: Minimum 4GB (8GB recommended for large images)
- **Storage**: At least 1GB free space
- **OS**: Windows 10+, macOS 10.14+, or Ubuntu 18.04+

## 🔧 Installation & Setup

### Step 1: Clone the Repository
```bash
git clone https://github.com/Utkarshjha09/Invisible-Watermarking.git
cd Invisible-Watermarking
```

### Step 2: Backend Setup (Flask API)

#### 2.1 Navigate to Backend Directory
```bash
cd backend
```

#### 2.2 Create Virtual Environment (Recommended)
```bash
# Create virtual environment
python -m venv .venv

# Activate virtual environment
# Windows:
.venv\Scripts\activate
# macOS/Linux:
source .venv/bin/activate
```

#### 2.3 Configure Environment Variables
```bash
# Copy example environment file
cp .env.example .env

# Edit .env file with your preferred settings
# Windows:
notepad .env
# macOS/Linux:
nano .env
```

**Key environment variables to configure:**
- `SECRET_KEY`: Change to a secure random string for production
- `CORS_ORIGINS`: Add your frontend URL for production
- `MAX_CONTENT_LENGTH`: Adjust file size limit as needed
- `FREQ_ALPHA`, `BLOCK_DCT_ALPHA`, `DEFAULT_DELTA`: Tune algorithm parameters

#### 2.4 Install Dependencies
```bash
# Upgrade pip first
python -m pip install --upgrade pip

# Install required packages
pip install -r requirements.txt
```

#### 2.5 Start Backend Server
```bash
python app.py
```
✅ **Backend server will start on `http://localhost:5000`**

### Step 3: Frontend Setup (React App)

#### 3.1 Open New Terminal
Keep the backend terminal running and open a new terminal window.

#### 3.2 Navigate to Frontend Directory
```bash
cd watermark-app
```

#### 3.3 Configure Environment Variables
```bash
# Copy example environment file
cp .env.example .env

# Edit .env file with your settings
# Windows:
notepad .env
# macOS/Linux:
nano .env
```

**Key frontend environment variables:**
- `VITE_API_URL`: Backend API URL (update for production)
- `VITE_MAX_FILE_SIZE`: Maximum file upload size
- `VITE_DEFAULT_ALGORITHM`: Default watermarking algorithm
- `VITE_ENABLE_3D_BACKGROUND`: Toggle 3D background animation

#### 3.4 Install Dependencies
```bash
# Install npm packages
npm install
```

#### 3.4 Start Development Server
```bash
npm run dev
```
✅ **Frontend will start on `http://localhost:5173`**

### Step 4: Access the Application
Open your web browser and navigate to `http://localhost:5173`
## 📖 How to Use the Application

### 🎯 **Embedding a Watermark**

#### Step 1: Access Embed Tab
1. Open the application at `http://localhost:5173`
2. Ensure you're on the "Embed Watermark" tab (default)

#### Step 2: Select Algorithm & Parameters
1. **Algorithm Selection**: Choose from the dropdown:
   - **Quantized Block DCT** (Recommended): Most accurate, bit-level embedding
   - **Block DCT**: Standard amplitude-based embedding  
   - **Frequency Domain**: FFT-based for specialized cases

2. **Delta Strength** (for Quantized): Adjust slider (0.02-0.10)
   - Lower values: Less visible, may be less robust
   - Higher values: More robust, slightly more visible
   - Recommended: 0.04-0.06

#### Step 3: Choose Watermark Type
- **Text Watermark**: Enter text in the input field
- **Image Watermark**: Upload a simple, high-contrast image (logos work best)

#### Step 4: Upload Host Image
1. **Drag & Drop**: Drag your image onto the upload area
2. **Browse**: Click the upload area to select a file
3. **Supported Formats**: PNG, JPG, JPEG
4. **Size Recommendations**: 
   - Minimum: 512×512 pixels for best results
   - Maximum: 2048×2048 pixels (larger files process slower)

#### Step 5: Process & Download
1. Click the **"Embed Watermark"** button
2. Wait for processing (progress indicator shows status)
3. Watermarked image downloads automatically
4. **File Naming**: Original filename with `_watermarked` suffix

### 🕵️ **Extracting a Watermark**

#### Step 1: Access Extract Tab
1. Click on the "Extract Watermark" tab
2. Interface switches to extraction mode

#### Step 2: Upload Watermarked Image
1. **Drag & Drop** or **Browse** to select the watermarked image
2. Must be the same format as the embedded image
3. Avoid re-compressed or heavily edited images for best results

#### Step 3: Choose Extraction Type
- **Auto** (Recommended): Attempts to detect watermark type automatically
- **Text**: Extract text watermarks only
- **Image**: Extract image watermarks only

#### Step 4: Extract & View Results
1. Click **"Extract Watermark"** button
2. **For Text**: Extracted text appears in the result area
3. **For Images**: Extracted image downloads automatically
4. **Success Indicators**: Clear, recognizable output
5. **Failure Signs**: Noisy, fragmented, or empty results

### 💡 **Best Practices**

#### **For Optimal Results**:

1. **Image Quality**:
   - Use high-quality, uncompressed images as hosts
   - Avoid heavily textured or noisy backgrounds
   - Smooth gradients and solid colors work best

2. **Watermark Design** (for image watermarks):
   - Simple, bold designs with high contrast
   - Avoid thin lines or small text
   - Black and white logos perform better than grayscale
   - Recommended size: 64×64 pixels or larger before embedding

3. **Algorithm Selection**:
   - **Quantized Block DCT**: Best for accuracy and robustness
   - **Block DCT**: Good balance of speed and quality
   - **Frequency Domain**: For specialized applications or research

4. **Parameter Tuning**:
   - Start with default delta (0.05)
   - Increase if extraction fails, decrease if embedding is visible
   - Test extraction immediately after embedding

#### **Troubleshooting Common Issues**:

1. **"No watermark found"**:
   - Try different extraction types (auto → text → image)
   - Re-embed with higher delta strength
   - Ensure image hasn't been compressed or edited

2. **Partial/Noisy Extraction**:
   - Original watermark may have been too detailed
   - Try re-embedding with simpler watermark
   - Use Quantized Block DCT algorithm

3. **Visible Watermark Artifacts**:
   - Reduce delta strength
   - Choose simpler watermark design
   - Use Block DCT instead of Frequency Domain

## 🧠 Technical Deep Dive

### 🔬 **Watermarking Algorithms Explained**

#### **1. Quantized Block DCT (Recommended)**
**How it works**: 
- Divides image into 8×8 blocks
- Applies DCT to each block
- Embeds watermark bits by ordering coefficient pairs
- Uses quantization for reliable bit recovery

**Advantages**:
- Deterministic bit-level embedding
- High extraction accuracy
- Robust against compression
- Clear binary output

**Best for**: Most applications requiring reliable watermark recovery

```python
# Pseudocode for Quantized Block DCT
for each 8x8 block:
    dct_block = dct2(block)
    if watermark_bit == 1:
        ensure dct_block[1,2] > dct_block[2,1] + delta
    else:
        ensure dct_block[1,2] < dct_block[2,1] - delta
```

#### **2. Block DCT (Standard)**
**How it works**:
- Similar block-wise processing
- Modulates coefficient amplitudes based on watermark
- Uses scaling factor (alpha) for embedding strength

**Advantages**:
- Fast processing
- Good invisibility
- Established method

**Best for**: Standard applications where speed is important

#### **3. Frequency Domain (FFT-based)**
**How it works**:
- Transforms entire image to frequency domain
- Embeds watermark in mid-frequency ring
- Uses magnitude modulation

**Advantages**:
- Global frequency analysis
- Good for textured images
- Research applications

**Best for**: Specialized use cases and research

### 📊 **Debug Information & Metrics**

When extracting watermarks, the backend logs detailed metrics:

```
[ExtractQBlockDCT] blocks=3136 bits=3136 valid=3089 accuracy=98.5%
[ExtractFreqScaled] amplify=31.25 mean=0.0064 std=0.0122 high=0.0371
```

**Understanding the Metrics**:
- **blocks**: Number of 8×8 blocks processed
- **bits**: Total watermark bits extracted
- **valid**: Successfully decoded bits
- **accuracy**: Percentage of reliable extractions
- **amplify**: Signal amplification factor used
- **mean/std/high**: Statistical measures of watermark signal strength

### 🔧 **Advanced Configuration**

#### **Backend Parameters** (in `app.py`):
```python
FREQ_ALPHA = 0.08          # Frequency domain embedding strength
BLOCK_DCT_ALPHA = 0.25     # Block DCT amplitude scaling
DEFAULT_DELTA = 0.05       # Quantized method default strength
MAX_WATERMARK_SIZE = 64    # Maximum watermark dimensions
```

#### **Frontend Customization**:
```javascript
// API endpoints configuration
const API_BASE_URL = 'http://localhost:5000';
const MAX_FILE_SIZE = 16 * 1024 * 1024; // 16MB limit
const SUPPORTED_FORMATS = ['image/png', 'image/jpeg', 'image/jpg'];
```

## 🚀 Production Deployment

### **Frontend Deployment (Netlify/Vercel)**

#### Build for Production:
```bash
cd watermark-app
npm run build
```

#### Deploy to Netlify:
```bash
# Install Netlify CLI
npm install -g netlify-cli

# Deploy
netlify deploy --prod --dir=dist
```

#### Environment Variables:
```bash
# Set API URL for production
VITE_API_URL=https://your-backend-domain.com
```

### **Backend Deployment (Heroku/Railway)**

#### Prepare for Deployment:
```bash
# Create Procfile
echo "web: gunicorn app:app" > Procfile

# Update requirements.txt
pip freeze > requirements.txt
```

#### Deploy to Heroku:
```bash
# Install Heroku CLI and login
heroku create your-watermark-api

# Deploy
git add .
git commit -m "Deploy to production"
git push heroku main
```

#### Environment Configuration:
```bash
# Set production environment variables
heroku config:set FLASK_ENV=production
heroku config:set MAX_CONTENT_LENGTH=16777216
```

### **Docker Deployment**

#### Frontend Dockerfile:
```dockerfile
FROM node:18-alpine
WORKDIR /app
COPY package*.json ./
RUN npm install
COPY . .
RUN npm run build
EXPOSE 5173
CMD ["npm", "run", "preview"]
```

#### Backend Dockerfile:
```dockerfile
FROM python:3.9-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY . .
EXPOSE 5000
CMD ["gunicorn", "--bind", "0.0.0.0:5000", "app:app"]
```

#### Docker Compose:
```yaml
version: '3.8'
services:
  frontend:
    build: ./watermark-app
    ports:
      - "5173:5173"
    depends_on:
      - backend
  
  backend:
    build: ./backend
    ports:
      - "5000:5000"
    environment:
      - FLASK_ENV=production
```

## 🐛 Troubleshooting Guide

### **Installation Issues**

#### Python Environment Problems:
```bash
# If Python packages fail to install
python -m pip install --upgrade pip setuptools wheel

# For OpenCV issues on Windows
pip install opencv-python-headless

# For M1 Mac compatibility
pip install --upgrade pip
pip install opencv-python --no-cache-dir
```

#### Node.js Issues:
```bash
# Clear npm cache
npm cache clean --force

# Delete node_modules and reinstall
rm -rf node_modules package-lock.json
npm install

# Use specific Node version with nvm
nvm install 18
nvm use 18
```

### **Runtime Issues**

#### Backend Server Won't Start:
1. **Check Python version**: `python --version` (requires 3.8+)
2. **Verify virtual environment**: Ensure activated with `which python`
3. **Check port availability**: Ensure port 5000 isn't in use
4. **Review error logs**: Look for specific error messages

#### Frontend Build Errors:
1. **Check Node version**: `node --version` (requires 16+)
2. **Clear Vite cache**: `rm -rf node_modules/.vite`
3. **Update dependencies**: `npm update`

#### CORS Errors:
```python
# Update backend CORS configuration
from flask_cors import CORS
CORS(app, origins=['http://localhost:5173', 'https://yourdomain.com'])
```

### **Watermarking Issues**

#### Poor Extraction Quality:
1. **Algorithm mismatch**: Ensure same algorithm for embed/extract
2. **Parameter tuning**: Adjust delta strength
3. **Image quality**: Use high-quality, uncompressed source images
4. **Watermark simplification**: Use simpler, high-contrast designs

#### Processing Failures:
1. **File format**: Ensure supported formats (PNG, JPG, JPEG)
2. **File size**: Check size limits (16MB default)
3. **Memory issues**: Reduce image size or restart backend
4. **Encoding issues**: Save images in standard color spaces

## 🤝 Contributing

### **Development Setup**
1. Fork the repository
2. Create feature branch: `git checkout -b feature-name`
3. Follow coding standards and add tests
4. Submit pull request with detailed description

### **Code Style**
- **Python**: Follow PEP 8 guidelines
- **JavaScript**: Use ESLint configuration
- **CSS**: BEM methodology for class naming
- **Comments**: Document complex algorithms and API endpoints

### **Testing**
```bash
# Backend tests
cd backend
python -m pytest test_watermarking.py

# Frontend tests
cd watermark-app
npm run test
```

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 👏 Acknowledgments

- **OpenCV**: Computer vision and image processing capabilities
- **React**: Modern UI framework for responsive interfaces  
- **Flask**: Lightweight backend API framework
- **OGL**: WebGL library for 3D graphics rendering
- **Vite**: Fast build tool for modern web development

## 📞 Support

For questions, issues, or contributions:

- **GitHub Issues**: [Report bugs or request features](https://github.com/Utkarshjha09/Invisible-Watermarking/issues)
- **Documentation**: Check this README for detailed information
- **Email**: Contact the maintainer for urgent issues

---

**Made with ❤️ by [Utkarshjha09](https://github.com/Utkarshjha09)**

*Invisible Watermarking Tool - Protecting digital content with advanced steganography techniques.*
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
