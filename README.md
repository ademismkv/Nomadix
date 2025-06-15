# Nomadix - Kyrgyz Ornament Analysis Platform

Nomadix is an AI-powered platform for analyzing and understanding Kyrgyz ornaments using computer vision and natural language processing. The platform can detect ornaments in images and provide detailed cultural explanations of their meanings.

## 🌟 Features

- **AI-Powered Ornament Detection**: Uses YOLOv8 to detect Kyrgyz ornaments in images
- **Cultural Analysis**: Provides detailed explanations of ornament meanings in multiple languages
- **Multi-language Support**: Analysis available in English, Russian, and Kyrgyz
- **Real-time Processing**: Fast and efficient image analysis
- **User-friendly Interface**: Modern, responsive design with intuitive controls

## 🚀 Tech Stack

### Backend
- FastAPI (Python web framework)
- YOLOv8 (Object detection)
- Grok API (Natural language processing)
- OpenCV (Image processing)
- Docker (Containerization)

### Frontend
- React
- TypeScript
- Tailwind CSS
- Shadcn UI Components

## 📋 Prerequisites

- Python 3.11+
- Node.js 18+
- Docker (for containerized deployment)
- Grok API key

## 🛠️ Installation

### Backend Setup

1. Clone the repository:
```bash
git clone https://github.com/ademismkv/Nomadix.git
cd Nomadix
```

2. Create and activate a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Create a `.env` file:
```env
GROK_API_KEY=your_grok_api_key
```

### Frontend Setup

1. Navigate to the frontend directory:
```bash
cd frontend
```

2. Install dependencies:
```bash
npm install
```

3. Create a `.env` file:
```env
REACT_APP_API_URL=https://nomadix-production.up.railway.app
```

## 🚀 Running the Application

### Development Mode

1. Start the backend:
```bash
uvicorn ornament_analyzer:app --reload
```

2. Start the frontend:
```bash
npm run dev
```

### Docker Deployment

1. Build the Docker image:
```bash
docker build -t nomadix .
```

2. Run the container:
```bash
docker run -p 8000:8000 nomadix
```

## 🌐 API Endpoints

### POST /analyze-image
Analyzes an image for Kyrgyz ornaments.

**Request:**
- Method: POST
- Content-Type: multipart/form-data
- Body: image file

**Response:**
```json
{
    "detected_ornaments": ["ornament1", "ornament2"],
    "ornament_counts": {
        "ornament1": 2,
        "ornament2": 1
    },
    "response": "Cultural analysis of the detected ornaments"
}
```

## 📦 Project Structure

```
nomadix/
├── ornament_analyzer.py    # FastAPI application
├── requirements.txt        # Python dependencies
├── Dockerfile             # Docker configuration
├── src/                   # Frontend source code
│   ├── components/        # React components
│   ├── config/           # Configuration files
│   └── contexts/         # React contexts
└── templates/            # HTML templates
```

## 🔧 Environment Variables

### Backend
- `GROK_API_KEY`: Your Grok API key for natural language processing

### Frontend
- `REACT_APP_API_URL`: URL of the backend API

## 🚢 Deployment

The application is configured for deployment on Railway:

1. Connect your GitHub repository to Railway
2. Set up the required environment variables
3. Deploy the application

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 👥 Authors

- Ademismkv - Initial work

## 🙏 Acknowledgments

- YOLOv8 team for the object detection model
- Grok API for natural language processing
- The Kyrgyz cultural heritage community

## 📞 Support

For support, please open an issue in the GitHub repository or contact the maintainers.

## 🔄 Updates

- Latest update: Added multi-language support
- Improved model accuracy
- Enhanced error handling
- Added Docker support 