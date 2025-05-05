/**
 * Phishing Website Analysis Chrome Extension
 * Author: Sita Vaibhavi Gunturi
 * 
 * This extension analyzes websites for potential phishing attempts using AI.
 * Note: API key has been removed for security as this is now a public repository.
 * 
 * Functions:
 * 1. analyzeWithAI(pageData): Analyzes website content using Google's Gemini API
 *    - Takes page data including URL, title, content, forms, images, and links
 *    - Returns analysis with rating (1-5 stars), risk level, and explanation
 * 
 * 2. Message Listener: Handles communication between popup and background script
 *    - Listens for 'analyzeWithAI' action
 *    - Processes website data and returns AI analysis results
 */

const API_KEY = process.env.GOOGLE_API_KEY;

chrome.runtime.onMessage.addListener((request, sender, sendResponse) => {
  if (request.action === 'analyzeWithAI') {
    analyzeWithAI(request.data).then(sendResponse).catch(error => {
      console.error('Error in background script:', error);
      sendResponse({ error: error.message });
    });
    return true; // Required for async response
  }
});

async function analyzeWithAI(pageData) {
  try {
    const prompt = `Analyze this website for potential phishing attempts. Consider the following:
    1. URL structure and domain
    2. Content and text
    3. Forms and input fields
    4. Images and their context
    5. Links and their destinations
    
    Website data:
    URL: ${pageData.url}
    Title: ${pageData.title}
    Content: ${pageData.content.substring(0, 1000)}...
    Forms: ${JSON.stringify(pageData.forms)}
    Images: ${JSON.stringify(pageData.images)}
    Links: ${JSON.stringify(pageData.links)}
    
    Provide a brief analysis in the following format:
    Rating: [1-5 stars]
    Risk Level: [Low/Medium/High]
    Brief Explanation: [2-3 sentences maximum]
    
    Keep the response concise and focused on key security concerns.`;

    console.log('Sending request to API...');
    const response = await fetch('https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent', {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        'x-goog-api-key': API_KEY
      },
      body: JSON.stringify({
        contents: [{
          parts: [{
            text: prompt
          }]
        }]
      })
    });

    console.log('Response status:', response.status);
    const responseText = await response.text();
    console.log('Raw response:', responseText);

    if (!response.ok) {
      throw new Error(`API request failed with status ${response.status}: ${responseText}`);
    }

    let data;
    try {
      data = JSON.parse(responseText);
    } catch (e) {
      throw new Error(`Failed to parse API response: ${e.message}`);
    }

    console.log('Parsed response:', data);

    if (!data.candidates || !data.candidates[0] || !data.candidates[0].content) {
      throw new Error('Invalid response format from API: ' + JSON.stringify(data));
    }

    const analysis = data.candidates[0].content.parts[0].text;
    console.log('Analysis result:', analysis);
    
    const ratingMatch = analysis.match(/Rating:\s*(\d+)/i);
    const riskMatch = analysis.match(/Risk Level:\s*([A-Za-z]+)/i);
    const explanationMatch = analysis.match(/Brief Explanation:\s*([^]*?)(?=\n|$)/i);
    
    const rating = ratingMatch ? parseInt(ratingMatch[1]) : 3;
    // Determine risk level based on rating
    let riskLevel;
    if (rating <= 2) {
      riskLevel = 'High';
    } else if (rating <= 4) {
      riskLevel = 'Medium';
    } else {
      riskLevel = 'Low';
    }
    const explanation = explanationMatch ? explanationMatch[1].trim() : 'Unable to analyze website.';
    
    return {
      rating: rating,
      riskLevel: riskLevel,
      explanation: explanation
    };
  } catch (error) {
    console.error('Error in AI analysis:', error);
    return {
      rating: 3,
      riskLevel: 'Medium',
      explanation: `Error analyzing website: ${error.message}`
    };
  }
} 