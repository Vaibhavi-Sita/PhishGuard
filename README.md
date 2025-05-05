# Phishing Website Detection using Generative AI

## Overview
This project implements an advanced phishing website detection system that combines traditional machine learning approaches with cutting-edge generative AI techniques. The system includes both a Chrome extension for real-time detection and a comprehensive evaluation framework for model performance analysis.

## Project Structure

```
├── PhishingWebsiteGenAI/          # Chrome Extension Implementation
│   ├── background.js              # Background service worker
│   ├── content.js                 # Content script for webpage analysis
│   ├── popup.js                   # Extension popup functionality
│   ├── manifest.json              # Extension configuration
│   ├── popup.html                 # Extension popup interface
│   └── icons/                     # Extension icons
│
├── finetuning and evaluation/     # Model Training and Evaluation
│   ├── combined_models.py         # Ensemble model implementation
│   ├── urlnet.py                  # URLNet model implementation
│   ├── logistic_regression.py     # Traditional ML model
│   ├── gemini.py                  # Gemini AI integration
│   └── requirements.txt           # Python dependencies
│
└── Datasets/                      # Training and testing datasets
```

## Features

### Chrome Extension
- Real-time phishing website detection
- User-friendly popup interface
- Background monitoring of web navigation
- Instant alerts for suspicious websites

### Machine Learning Models
- Ensemble approach combining multiple detection methods
- URLNet implementation for deep learning-based detection
- Traditional logistic regression model
- Integration with Google's Gemini AI for advanced analysis

## Installation

### Chrome Extension
1. Clone this repository
2. Open Chrome and navigate to `chrome://extensions/`
3. Enable "Developer mode"
4. Click "Load unpacked" and select the `PhishingWebsiteGenAI` directory

### Development Environment
1. Create a virtual environment:
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate
   ```
2. Install dependencies:
   ```bash
   cd "finetuning and evaluation"
   pip install -r requirements.txt
   ```

## Usage

### Chrome Extension
1. Click the extension icon in your browser toolbar
2. The popup will show the current website's security status
3. Navigate to any website to get real-time phishing detection results

### Model Training and Evaluation
1. Navigate to the `finetuning and evaluation` directory
2. Run individual models:
   ```bash
   python logistic_regression.py
   python urlnet.py
   python gemini.py
   ```
3. For ensemble model:
   ```bash
   python combined_models.py
   ```

## Technical Details

### Models
- **Logistic Regression**: Traditional ML approach for baseline performance
- **URLNet**: Deep learning model for URL analysis
- **Gemini AI**: Advanced language model integration
- **Ensemble Model**: Combines predictions from all models for improved accuracy

### Extension Components
- **Background Script**: Handles URL monitoring and model inference
- **Content Script**: Analyzes webpage content
- **Popup Interface**: Displays detection results and user controls

## Contributing
Contributions are welcome! Please feel free to submit a Pull Request.

## Acknowledgments
- Google Gemini AI for advanced language model capabilities
- URLNet authors for the deep learning architecture
- Chrome Extension API documentation 
