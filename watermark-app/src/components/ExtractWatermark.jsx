import { useState } from 'react'
import axios from 'axios'
import FileUpload from './FileUpload'
import { TrialButton } from './TrialButton'

const ExtractWatermark = () => {
  const [originalFile, setOriginalFile] = useState(null)
  const [watermarkedFile, setWatermarkedFile] = useState(null)
  const [extractionType, setExtractionType] = useState('image') // Default to 'image' since most users test with images
  const [loading, setLoading] = useState(false)
  const [result, setResult] = useState(null)
  const [error, setError] = useState(null)

  const handleSubmit = async (e) => {
    e.preventDefault()
    
    if (!originalFile || !watermarkedFile) {
      setError('Please select both the original and watermarked images')
      return
    }

    setLoading(true)
    setError(null)
    setResult(null)

    try {
      const formData = new FormData()
      formData.append('original', originalFile)
      formData.append('watermarked', watermarkedFile)
      formData.append('type', extractionType)

      if (extractionType === 'text') {
        const response = await axios.post('http://localhost:5000/extract', formData, {
          headers: {
            'Content-Type': 'multipart/form-data',
          },
          responseType: 'json' // Expect JSON response with extracted text
        })

        // Display the extracted watermark text
        if (response.data && response.data.watermark) {
          setResult(`Extracted watermark: "${response.data.watermark}"`)
        } else {
          setError('No watermark text could be extracted from the image. Make sure the image was watermarked with text, not an image.')
        }
      } else {
        const response = await axios.post('http://localhost:5000/extract', formData, {
          headers: {
            'Content-Type': 'multipart/form-data',
          },
          responseType: 'blob' // Expect file download for image
        })

        // Create a download link for the extracted image
        const url = window.URL.createObjectURL(new Blob([response.data]))
        const link = document.createElement('a')
        link.href = url
        link.setAttribute('download', 'extracted_watermark.png')
        document.body.appendChild(link)
        link.click()
        link.remove()
        window.URL.revokeObjectURL(url)

        setResult('Watermark image extracted successfully! Download started.')
      }
    } catch (err) {
      console.error('Error extracting watermark:', err)
      if (err.response && err.response.data) {
        if (err.response.data.error) {
          setError(err.response.data.error)
        } else {
          setError(`Extraction failed. Make sure you selected the correct extraction type (${extractionType}).`)
        }
      } else {
        setError('Network error. Please make sure the backend server is running.')
      }
    } finally {
      setLoading(false)
    }
  }

  const resetForm = () => {
    setOriginalFile(null)
    setWatermarkedFile(null)
    setResult(null)
    setError(null)
  }

  return (
    <div className="extract-watermark">
      <h2>Extract Watermark</h2>
      <p>Upload the original image and the watermarked image to extract the embedded content.</p>
      
      <form onSubmit={handleSubmit} className="watermark-form">
        {/* Extraction Type Selector */}
        <div className="extraction-type-selector">
          <h3>Choose Extraction Type</h3>
          <div className="radio-group">
            <label className="radio-option">
              <input
                type="radio"
                value="text"
                checked={extractionType === 'text'}
                onChange={(e) => setExtractionType(e.target.value)}
              />
              <span className="radio-label">Extract Text</span>
            </label>
            <label className="radio-option">
              <input
                type="radio"
                value="image"
                checked={extractionType === 'image'}
                onChange={(e) => setExtractionType(e.target.value)}
              />
              <span className="radio-label">Extract Image</span>
            </label>
            <label className="radio-option">
              <input
                type="radio"
                value="auto"
                checked={extractionType === 'auto'}
                onChange={(e) => setExtractionType(e.target.value)}
              />
              <span className="radio-label">Auto-Detect</span>
            </label>
          </div>
        </div>

        <div className="upload-sections-container">
          <div className="upload-section">
            <h3>Select Original Image</h3>
            <FileUpload
              file={originalFile}
              setFile={setOriginalFile}
              accept="image/*"
              placeholder="Drop your original image here or click to browse"
            />
          </div>

          <div className="upload-section">
            <h3>Select Watermarked Image</h3>
            <FileUpload
              file={watermarkedFile}
              setFile={setWatermarkedFile}
              accept="image/*"
              placeholder="Drop your watermarked image here or click to browse"
            />
          </div>
        </div>

        <div className="form-actions">
          <TrialButton 
            type="submit" 
            disabled={loading || !originalFile || !watermarkedFile}
          >
            {loading ? 'Processing...' : 'Extract Watermark'}
          </TrialButton>
          <TrialButton 
            type="button" 
            onClick={resetForm}
          >
            Reset
          </TrialButton>
        </div>
      </form>

      {error && (
        <div className="error-message">
          <p>{error}</p>
        </div>
      )}

      {result && (
        <div className="success-message">
          <p>{result}</p>
        </div>
      )}

      {loading && (
        <div className="loading-overlay">
          <div className="loader"></div>
          <p>Extracting watermark...</p>
        </div>
      )}
    </div>
  )
}

export default ExtractWatermark
