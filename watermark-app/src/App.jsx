/**
 * Main Application Component
 * 
 * This is the root component of the Invisible Watermarking Tool.
 * It handles navigation between different views (Home, Embed, Extract)
 * and manages the overall application state.
 */

// React hooks and utilities
import { useState } from 'react'
import './App.css'

// Component imports
import EmbedWatermark from './components/EmbedWatermark'  // Component for embedding watermarks
import ExtractWatermark from './components/ExtractWatermark'  // Component for extracting watermarks
import Prism from './components/Prism'  // 3D animated background component
import { TrialButton } from './components/TrialButton'  // Custom animated button component

function App() {
  // State management for navigation tabs
  // 'home', 'embed', or 'extract'
  const [activeTab, setActiveTab] = useState('home')

  /**
   * Content Renderer Function
   * Conditionally renders different components based on the active tab
   * @returns {JSX.Element} The appropriate component for the current tab
   */
  const renderContent = () => {
    switch (activeTab) {
      case 'embed':
        return <EmbedWatermark />
      case 'extract':
        return <ExtractWatermark />
      default:
        // Home page content with navigation buttons and information
        return (
          <div className="home-content">
            <h1>Invisible Watermarking Tool</h1>
            <p>Choose an operation to get started:</p>
            
            {/* Main action buttons */}
            <div className="button-group">
              <TrialButton 
                onClick={() => setActiveTab('embed')}
              >
                Embed Watermark
              </TrialButton>
              <TrialButton 
                onClick={() => setActiveTab('extract')}
              >
                Extract Watermark
              </TrialButton>
            </div>
            
            {/* Information section about the tool */}
            <div className="description">
              <h3>About Invisible Watermarking</h3>
              <p>
                This tool uses Discrete Cosine Transform (DCT) to embed invisible watermarks 
                into images. The watermark is embedded in the frequency domain, making it 
                robust and difficult to detect.
              </p>
              <ul>
                <li><strong>Embed:</strong> Add an invisible watermark to your image</li>
                <li><strong>Extract:</strong> Recover the watermark from a watermarked image</li>
              </ul>
            </div>
          </div>
        )
    }
  }

  return (
    <div className="app">
      {/* 
        3D Animated Background Component
        Creates a beautiful prism-like animated background with customizable properties
      */}
      <Prism
        animationType="3drotate"    // Type of animation
        height={4}                  // Prism height
        baseWidth={6}              // Base width of the prism
        glow={2}                   // Glow intensity
        noise={0.3}                // Noise factor for texture
        scale={3}                  // Overall scale
        hueShift={0.5}             // Color hue shifting
        colorFrequency={1.2}       // Color change frequency
        timeScale={0.4}            // Animation speed
        bloom={2}                  // Bloom effect intensity
        transparent={true}         // Enable transparency
        suspendWhenOffscreen={false} // Keep animating when not visible
      />
      
      {/* 
        Navigation Bar
        Fixed header with brand logo and navigation tabs
      */}
      <nav className="navbar">
        <div className="nav-container">
          {/* Brand Section with Custom SVG Logo */}
          <div className="nav-brand">
            <div className="brand-logo">
              {/* Custom SVG logo representing a figure walking on water */}
              <svg width="36" height="36" viewBox="0 0 100 100" xmlns="http://www.w3.org/2000/svg">
                {/* Gradient definitions for colors */}
                <defs>
                  <linearGradient id="logoGrad" x1="0%" y1="0%" x2="100%" y2="100%">
                    <stop offset="0%" stopColor="#4FC2FE"/>
                    <stop offset="50%" stopColor="#2E86C1"/>
                    <stop offset="100%" stopColor="#1B4F72"/>
                  </linearGradient>
                  <radialGradient id="bgGrad" cx="50%" cy="50%" r="50%">
                    <stop offset="0%" stopColor="#E8F6FF"/>
                    <stop offset="100%" stopColor="#B3E5FC"/>
                  </radialGradient>
                </defs>
                
                {/* Circular background for the logo */}
                <circle cx="50" cy="50" r="48" fill="url(#bgGrad)" stroke="url(#logoGrad)" strokeWidth="2"/>
                
                {/* Water wave patterns */}
                <path d="M10 70 Q30 65 50 70 T90 70" fill="none" stroke="url(#logoGrad)" strokeWidth="2" opacity="0.6"/>
                <path d="M15 75 Q35 70 55 75 T85 75" fill="none" stroke="url(#logoGrad)" strokeWidth="1.5" opacity="0.4"/>
                
                {/* Walking figure silhouette */}
                <g transform="translate(45, 30)">
                  {/* Head */}
                  <circle cx="5" cy="5" r="4" fill="url(#logoGrad)"/>
                  {/* Body */}
                  <rect x="2" y="9" width="6" height="15" rx="3" fill="url(#logoGrad)"/>
                  {/* Arms in walking position */}
                  <rect x="-1" y="12" width="4" height="2" rx="1" fill="url(#logoGrad)" transform="rotate(-20)"/>
                  <rect x="7" y="12" width="4" height="2" rx="1" fill="url(#logoGrad)" transform="rotate(20)"/>
                  {/* Legs in walking position */}
                  <rect x="3" y="24" width="2" height="8" rx="1" fill="url(#logoGrad)" transform="rotate(-10)"/>
                  <rect x="5" y="24" width="2" height="8" rx="1" fill="url(#logoGrad)" transform="rotate(10)"/>
                </g>
                
                {/* Water ripples around feet */}
                <circle cx="45" cy="75" r="3" fill="none" stroke="url(#logoGrad)" strokeWidth="1" opacity="0.3"/>
                <circle cx="45" cy="75" r="6" fill="none" stroke="url(#logoGrad)" strokeWidth="0.8" opacity="0.2"/>
                <circle cx="55" cy="77" r="2" fill="none" stroke="url(#logoGrad)" strokeWidth="0.8" opacity="0.25"/>
              </svg>
            </div>
            <h2>Invisible Watermarking</h2>
          </div>
          
          {/* Navigation Menu */}
          <div className="nav-menu">
            {/* Home Tab */}
            <button 
              className={`nav-item ${activeTab === 'home' ? 'active' : ''}`}
              onClick={() => setActiveTab('home')}
            >
              <span className="nav-icon">🏠</span>
              <span className="nav-text">Home</span>
            </button>
            
            {/* Embed Tab */}
            <button 
              className={`nav-item ${activeTab === 'embed' ? 'active' : ''}`}
              onClick={() => setActiveTab('embed')}
            >
              <span className="nav-icon">📝</span>
              <span className="nav-text">Embed</span>
            </button>
            
            {/* Extract Tab */}
            <button 
              className={`nav-item ${activeTab === 'extract' ? 'active' : ''}`}
              onClick={() => setActiveTab('extract')}
            >
              <span className="nav-icon">🔍</span>
              <span className="nav-text">Extract</span>
            </button>
          </div>
        </div>
      </nav>
      
      {/* 
        Main Content Area
        Renders different components based on the active tab
      */}
      <main className="main-content">
        {renderContent()}
      </main>
    </div>
  )
}

export default App
