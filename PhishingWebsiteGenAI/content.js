/**
 * Phishing Website Analysis Chrome Extension - Content Script
 * Author: Sita Vaibhavi Gunturi
 * 
 * This script runs in the context of web pages to collect data for phishing analysis.
 * 
 * Functions:
 * 1. analyzePage(): Collects and processes webpage data
 *    - Gathers URL, title, content, images, forms, and links
 *    - Sends data to background script for AI analysis
 *    - Returns analysis results or error message
 * 
 * 2. Message Listener: Handles communication with popup
 *    - Listens for 'analyze' action
 *    - Triggers page analysis and returns results
 */

// Listen for messages from the popup
chrome.runtime.onMessage.addListener((request, sender, sendResponse) => {
  console.log('Content script received message:', request);
  
  if (request.action === 'analyze') {
    analyzePage()
      .then(result => {
        console.log('Analysis complete, sending result:', result);
        sendResponse({ result: result });
      })
      .catch(error => {
        console.error('Error in content script:', error);
        sendResponse({ error: error.message });
      });
    return true; // Required for async response
  }
});

async function analyzePage() {
  try {
    console.log('Starting page analysis...');
    
    // Collect page data
    const pageData = {
      url: window.location.href,
      title: document.title,
      content: document.body.innerText,
      images: Array.from(document.images).map(img => ({
        src: img.src,
        alt: img.alt
      })),
      forms: Array.from(document.forms).map(form => ({
        action: form.action,
        method: form.method,
        inputs: Array.from(form.elements).map(input => ({
          type: input.type,
          name: input.name
        }))
      })),
      links: Array.from(document.links).map(link => ({
        href: link.href,
        text: link.textContent
      }))
    };

    console.log('Collected page data:', pageData);

    // Validate collected data
    if (!pageData.url || !pageData.title) {
      throw new Error('Failed to collect basic page information');
    }

    // Send data to background script for AI analysis
    console.log('Sending data to background script...');
    const response = await chrome.runtime.sendMessage({
      action: 'analyzeWithAI',
      data: pageData
    });

    console.log('Received response from background script:', response);

    if (response.error) {
      throw new Error(response.error);
    }

    return response;
  } catch (error) {
    console.error('Error analyzing page:', error);
    return { error: error.message };
  }
} 