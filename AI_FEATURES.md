# AI Features Documentation

## Overview
The Warren-DCA application now includes an AI Financial Assistant powered by Google Gemini AI. This feature provides intelligent stock analysis and investment advice based on Warren Buffett's principles.

## Key Features

### 🤖 AI Financial Assistant
- **Location**: Always accessible in the sidebar (outermost section)
- **Purpose**: Provides financial analysis and investment advice
- **Technology**: Google Gemini AI integration
- **Language**: Supports both English and Thai

### 📊 Context-Aware Analysis
The AI assistant automatically receives context about:
- Selected stocks and market
- DCA simulation settings
- Current analysis results
- Conversation history

### 💾 Database Storage
All AI interactions are stored in a SQLite database (`ai_queries.db`) with:
- Query and response content
- Timestamp and session tracking
- Context data for each interaction
- Search and management capabilities

### 🚀 No Page Refresh
- Uses Streamlit session state
- Maintains conversation continuity
- Preserves application state during AI queries
- Smooth user experience

## Setup Instructions

### 1. Get Google AI API Key
1. Visit [Google AI Studio](https://aistudio.google.com/)
2. Create an account and generate an API key
3. Keep the API key secure

### 2. Configure API Key
Choose one of these methods:

#### Option A: Environment Variable
```bash
export GOOGLE_AI_API_KEY="your_api_key_here"
```

#### Option B: .env File
Create a `.env` file in the project root:
```
GOOGLE_AI_API_KEY=your_api_key_here
```

#### Option C: Streamlit Secrets
Create `.streamlit/secrets.toml`:
```toml
GOOGLE_AI_API_KEY = "your_api_key_here"
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Run Application
```bash
streamlit run streamlit_app.py
```

## Usage Guide

### Basic Usage
1. Configure your Google AI API key (see setup above)
2. Select stocks for analysis on the main page
3. Use the AI assistant in the sidebar to ask questions
4. View conversation history in the "AI Chat History" page

### Sample Questions
The AI provides sample questions like:
- "Analyze the selected stocks based on Warren Buffett principles"
- "What are the key risks in my stock selection?"
- "How does the DCA simulation look for my portfolio?"
- "Which stock shows the best Buffett checklist score?"

### Features
- **Sample Questions**: Click pre-built questions for quick analysis
- **Context Awareness**: AI knows your current stock selections and settings
- **Conversation History**: Maintains chat history within session
- **Search**: Search through previous conversations
- **Data Management**: Clear old conversations or all data

## Database Schema

The AI database (`ai_queries.db`) contains:

```sql
CREATE TABLE ai_queries (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
    query TEXT NOT NULL,
    response TEXT NOT NULL,
    context_data TEXT,  -- JSON string
    session_id TEXT,
    user_id TEXT DEFAULT 'anonymous'
);
```

## File Structure

```
├── ai_helper.py          # Google Gemini AI integration
├── ai_database.py        # SQLite database management
├── streamlit_app.py      # Main application with AI features
├── ai_queries.db         # Database file (auto-created)
├── .env                  # Environment variables (optional)
└── requirements.txt      # Updated with AI dependencies
```

## API Dependencies

### New Dependencies Added:
- `google-generativeai`: Google Gemini AI integration
- `python-dotenv`: Environment variable loading

### Existing Dependencies:
- `streamlit`: Web application framework
- `yfinance`: Stock data
- `pandas`: Data manipulation
- `matplotlib`: Charts and visualization

## Security Notes

1. **API Key Security**: Never commit your Google AI API key to version control
2. **Database Security**: The `ai_queries.db` file contains user conversations and is gitignored
3. **Environment Variables**: Use secure methods to set environment variables in production

## Troubleshooting

### AI Not Working
1. Check if Google AI API key is configured correctly
2. Verify internet connection for API calls
3. Check API key permissions and quota

### Database Issues
1. Ensure write permissions in application directory
2. Check disk space for database file
3. Verify SQLite is available

### Performance
1. Database automatically manages old data (30+ days cleanup option)
2. Conversation history limited to recent messages for context
3. API calls are rate-limited by Google AI

## Integration with Existing Features

The AI features are fully integrated with the existing Warren-DCA functionality:
- Stock analysis results provide context to AI
- DCA simulation data informs AI recommendations
- Market selection affects AI advice
- All existing features remain unchanged and functional

## Future Enhancements

Potential improvements:
- Multi-language AI responses
- Advanced financial analysis prompts
- Integration with more AI providers
- Enhanced context processing
- User authentication and personalized history