// API Configuration
const API_URL = process.env.REACT_APP_API_URL || 'https://nomadix-production.up.railway.app';

export const config = {
    apiUrl: API_URL,
    endpoints: {
        analyzeImage: `${API_URL}/analyze-image`,
    }
}; 