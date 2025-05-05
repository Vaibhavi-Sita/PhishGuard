/**
 * Phishing Website Analysis Chrome Extension - Popup Script
 * Author: Sita Vaibhavi Gunturi
 * 
 * This script manages the extension's popup interface and user interactions.
 * 
 * Functions:
 * 1. analyzeDirectly(url): Fallback analysis function
 *    - Used when content script injection fails
 *    - Collects basic page data and sends to background script
 * 
 * 2. updateStatus(result): Updates the popup UI
 *    - Displays star rating, risk level, and explanation
 *    - Sets appropriate status class based on rating
 * 
 * 3. Event Listeners:
 *    - Analyzes current tab on button click
 *    - Handles content script injection
 *    - Manages communication with content script
 */

document.addEventListener('DOMContentLoaded', function() {
  const analyzeButton = document.getElementById('analyze');
  const statusDiv = document.getElementById('status');

  analyzeButton.addEventListener('click', async () => {
    statusDiv.textContent = 'Analyzing...';
    statusDiv.className = 'status';

    try {
      // Get the current active tab
      const [tab] = await chrome.tabs.query({ active: true, currentWindow: true });
      console.log('Current tab:', tab);

      // Check if we can inject the content script
      if (!tab.url.startsWith('chrome://') && !tab.url.startsWith('chrome-extension://')) {
        try {
          // Try to inject the content script if it's not already there
          await chrome.scripting.executeScript({
            target: { tabId: tab.id },
            files: ['content.js']
          });
        } catch (injectError) {
          console.log('Content script already injected or injection failed:', injectError);
        }
      }
      
      // Send message to content script to analyze the page
      chrome.tabs.sendMessage(tab.id, { action: 'analyze' }, (response) => {
        console.log('Received response:', response);
        
        if (chrome.runtime.lastError) {
          console.error('Runtime error:', chrome.runtime.lastError);
          // If we get a connection error, try to analyze directly from the popup
          if (chrome.runtime.lastError.message.includes('Receiving end does not exist')) {
            analyzeDirectly(tab.url).then(result => {
              updateStatus(result);
            }).catch(error => {
              statusDiv.textContent = 'Error: ' + error.message;
              statusDiv.className = 'status danger';
            });
          } else {
            statusDiv.textContent = 'Error: ' + chrome.runtime.lastError.message;
            statusDiv.className = 'status danger';
          }
          return;
        }

        if (!response) {
          console.error('No response received');
          statusDiv.textContent = 'Error: No response received';
          statusDiv.className = 'status danger';
          return;
        }

        if (response.error) {
          console.error('Error in response:', response.error);
          statusDiv.textContent = 'Error: ' + response.error;
          statusDiv.className = 'status danger';
          return;
        }

        if (response.result) {
          console.log('Analysis result:', response.result);
          updateStatus(response.result);
        } else {
          console.error('Invalid response format:', response);
          statusDiv.textContent = 'Error: Invalid response format';
          statusDiv.className = 'status danger';
        }
      });
    } catch (error) {
      console.error('Error in popup:', error);
      statusDiv.textContent = 'Error: ' + error.message;
      statusDiv.className = 'status danger';
    }
  });
});

async function analyzeDirectly(url) {
  // Fallback analysis when content script fails
  const pageData = {
    url: url,
    title: document.title,
    content: document.body.innerText,
    images: [],
    forms: [],
    links: []
  };

  // Send data to background script for AI analysis
  const response = await chrome.runtime.sendMessage({
    action: 'analyzeWithAI',
    data: pageData
  });

  return response;
}

function updateStatus(result) {
  console.log('Updating status with result:', result);
  const statusDiv = document.getElementById('status');
  const rating = result.rating;
  const riskLevel = result.riskLevel;
  const explanation = result.explanation;

  // Set status class based on rating
  if (rating >= 4) {
    statusDiv.className = 'status safe';
  } else if (rating >= 3) {
    statusDiv.className = 'status warning';
  } else {
    statusDiv.className = 'status danger';
  }

  // Create stars HTML using HTML entities
  const filledStar = '&starf;'; // Filled star entity
  const emptyStar = '&star;';   // Empty star entity
  const stars = filledStar.repeat(rating) + emptyStar.repeat(5 - rating);
  
  // Update status content
  statusDiv.innerHTML = `
    <div style="font-size: 24px; margin-bottom: 10px; color: #ffd700;">${stars}</div>
    <div style="font-weight: bold; margin-bottom: 5px;">Risk Level: ${riskLevel}</div>
    <div style="font-size: 12px;">${explanation}</div>
  `;
} 