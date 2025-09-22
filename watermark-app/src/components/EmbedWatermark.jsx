/**
 * EmbedWatermark Component
 * 
 * This component handles the watermark embedding functionality.
 * It allows users to upload an original image and a watermark image,
 * then sends them to the backend for DCT watermark embedding.
 */

// React hooks and HTTP client
import { useState } from 'react'
import axios from 'axios'

// Custom components
import FileUpload from './FileUpload'  // File upload component with drag & drop
import { TrialButton } from './TrialButton'  // Animated button component

const EmbedWatermark = () => {
  // State management for form data and UI states
  const [imageFile, setImageFile] = useState(null)       // Original image file
  const [watermarkText, setWatermarkText] = useState('') // Watermark text string
  const [watermarkFile, setWatermarkFile] = useState(null) // Watermark image file
  const [watermarkType, setWatermarkType] = useState('text') // 'text' or 'image'
  const [loading, setLoading] = useState(false)          // Loading state during processing
  const [algorithm, setAlgorithm] = useState('block_dct_q') // Embedding algorithm for image watermark
  const [delta, setDelta] = useState(0.04) // Strength parameter for quantized block DCT
  const [result, setResult] = useState(null)             // Success message
  const [error, setError] = useState(null)               // Error message

  /**
   * Form Submit Handler
   * Processes the watermark embedding request by sending files to the backend
   * @param {Event} e - Form submit event
   */
  const handleSubmit = async (e) => {
    e.preventDefault()
    
    // Validation: Check if image is provided
    if (!imageFile) {
      setError('Please select an image file')
      return
    }
    
    // Validation based on watermark type
    if (watermarkType === 'text') {
      if (!watermarkText.trim()) {
        setError('Please enter watermark text')
        return
      }
    } else if (watermarkType === 'image') {
      if (!watermarkFile) {
        setError('Please select a watermark image')
        return
      }
    }

    // Reset states and start loading
    setLoading(true)
    setError(null)
    setResult(null)

    try {
      // Prepare form data for multipart upload
      const formData = new FormData()
      formData.append('image', imageFile)
      
      if (watermarkType === 'text') {
        formData.append('watermark', watermarkText)
        formData.append('type', 'text')
      } else {
        formData.append('watermark', watermarkFile)
        formData.append('type', 'image')
        formData.append('algorithm', algorithm)
        if (algorithm === 'block_dct_q') {
          formData.append('delta', delta)
        }
      }

      // Send POST request to Flask backend
      const response = await axios.post('http://localhost:5000/embed', formData, {
        headers: {
          'Content-Type': 'multipart/form-data',
        },
        responseType: 'blob' // Important for file download
      })

      // Create a download link for the result
      const url = window.URL.createObjectURL(new Blob([response.data]))
      const link = document.createElement('a')
      link.href = url
      link.setAttribute('download', 'watermarked_image.png')
      document.body.appendChild(link)
      link.click()
      link.remove()
      window.URL.revokeObjectURL(url)

      setResult('Watermark embedded successfully! Download started.')
    } catch (err) {
      console.error('Error embedding watermark:', err)
      if (err.response && err.response.data) {
        // If the error response is a blob, convert it to text
        if (err.response.data instanceof Blob) {
          const text = await err.response.data.text()
          try {
            const errorData = JSON.parse(text)
            setError(errorData.error || 'Failed to embed watermark')
          } catch {
            setError('Failed to embed watermark')
          }
        } else {
          setError(err.response.data.error || 'Failed to embed watermark')
        }
      } else {
        setError('Network error. Please make sure the backend server is running.')
      }
    } finally {
      setLoading(false)
    }
  }

  const resetForm = () => {
    setImageFile(null)
    setWatermarkText('')
    setWatermarkFile(null)
    setResult(null)
    setError(null)
  }

  return (
    <div className="embed-watermark">
      <h2>Embed Watermark</h2>
      <p>Upload an image and choose between text or image watermark to create a watermarked image.</p>
      
      <form onSubmit={handleSubmit} className="watermark-form">
        <div className="upload-sections-container">
          <div className="upload-section">
            <h3>Select Image to Watermark</h3>
            <FileUpload
              file={imageFile}
              setFile={setImageFile}
              accept="image/*"
              placeholder="Drop your image here or click to browse"
            />
          </div>

          <div className="upload-section">
            <h3>Choose Watermark Type</h3>
            
            {/* Watermark Type Selector */}
            <div className="watermark-type-selector">
              <div className="radio-group">
                <label className="radio-option">
                  <input
                    type="radio"
                    value="text"
                    checked={watermarkType === 'text'}
                    onChange={(e) => setWatermarkType(e.target.value)}
                  />
                  <span className="radio-label">Text Watermark</span>
                </label>
                <label className="radio-option">
                  <input
                    type="radio"
                    value="image"
                    checked={watermarkType === 'image'}
                    onChange={(e) => setWatermarkType(e.target.value)}
                  />
                  <span className="radio-label">Image Watermark</span>
                </label>
              </div>
            </div>

            {/* Conditional Watermark Input */}
            {watermarkType === 'text' ? (
              <div className="text-input-container">
                <textarea
                  value={watermarkText}
                  onChange={(e) => setWatermarkText(e.target.value)}
                  placeholder="Enter the text you want to embed as watermark..."
                  className="watermark-text-input"
                  rows={3}
                  maxLength={100}
                />
                <div className="text-info">
                  <span className="char-count">{watermarkText.length}/100 characters</span>
                </div>
              </div>
            ) : (
              <div className="image-input-container">
                <FileUpload
                  file={watermarkFile}
                  setFile={setWatermarkFile}
                  accept="image/*"
                  placeholder="Drop your watermark image here or click to browse"
                />

                {/* Algorithm Selection */}
                <div className="algorithm-selector" style={{marginTop:'1rem'}}>
                  <h4 style={{marginBottom:'0.5rem'}}>Algorithm</h4>
                  <select
                    value={algorithm}
                    onChange={(e)=>setAlgorithm(e.target.value)}
                    className="algorithm-select"
                    style={{padding:'6px 10px', borderRadius:'6px'}}
                  >
                    <option value="block_dct_q">Block DCT Quantized (Recommended)</option>
                    <option value="block_dct">Block DCT (Amplitude)</option>
                    <option value="freq">Frequency Domain</option>
                  </select>
                  {algorithm === 'block_dct_q' && (
                    <div className="delta-control" style={{marginTop:'0.75rem'}}>
                      <label style={{display:'block', fontSize:'0.85rem', marginBottom:'0.25rem'}}>
                        Strength (delta): {delta.toFixed(3)}
                      </label>
                      <input
                        type="range"
                        min="0.02"
                        max="0.10"
                        step="0.005"
                        value={delta}
                        onChange={(e)=>setDelta(parseFloat(e.target.value))}
                        style={{width:'100%'}}
                      />
                      <small style={{display:'block', opacity:0.7, marginTop:'0.25rem'}}>
                        Higher = more robust extraction, but slightly more visible.
                      </small>
                    </div>
                  )}
                  {algorithm === 'freq' && (
                    <small style={{display:'block', opacity:0.7, marginTop:'0.5rem'}}>
                      Frequency embedding is subtle but harder to extract cleanly.
                    </small>
                  )}
                </div>
              </div>
            )}
          </div>
        </div>

        <div className="form-actions">
          <TrialButton 
            type="submit" 
            disabled={loading || !imageFile || (watermarkType === 'text' ? !watermarkText.trim() : !watermarkFile)}
          >
            {loading ? 'Processing...' : 'Embed Watermark'}
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
          <p>Processing watermark...</p>
        </div>
      )}
    </div>
  )
}

export default EmbedWatermark
