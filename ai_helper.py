import os
import google.genai as genai
from typing import Dict, Any, Optional, List
import streamlit as st
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

class AIHelper:
    """Helper class for AI queries using Google Gemini."""
    
    def __init__(self):
        self.model = None
        self.is_configured = False
        self.configure_gemini()
    
    def configure_gemini(self):
        """Configure Google Gemini with API key."""
        # Try to get API key from environment variable first
        api_key = os.getenv('GOOGLE_AI_API_KEY')
        
        if not api_key:
            # Try to get from Streamlit secrets
            try:
                api_key = st.secrets.get("GOOGLE_AI_API_KEY")
            except:
                pass
        
        if api_key:
            try:
                genai.configure(api_key=api_key)
                self.model = genai.GenerativeModel('gemini-2.5-flash')
                self.is_configured = True
            except Exception as e:
                st.warning(f"Failed to configure Gemini AI: {e}")
                self.is_configured = False
        else:
            self.is_configured = False
    
    def is_ready(self) -> bool:
        """Check if AI is ready to use."""
        return self.is_configured and self.model is not None
    
    def get_stock_analysis_prompt(self, stock_data: Dict[str, Any]) -> str:
        """Generate a context-rich prompt for stock analysis."""
        prompt = f"""
As a financial advisor with expertise in Warren Buffett's investment strategies, please analyze the following stock data:

Stock Symbol: {stock_data.get('symbol', 'N/A')}
Company Name: {stock_data.get('company_name', 'N/A')}

Financial Analysis Results:
- Buffett Checklist Score: {stock_data.get('buffett_score', 'N/A')} out of {stock_data.get('buffett_total', 'N/A')}
- Score Percentage: {stock_data.get('score_percentage', 'N/A')}%
- Badge Rating: {stock_data.get('badge', 'N/A')}

DCA Simulation Results:
- Total Investment: {stock_data.get('total_investment', 'N/A')}
- Current Value: {stock_data.get('current_value', 'N/A')}
- Profit/Loss: {stock_data.get('profit_loss', 'N/A')}
- Return Percentage: {stock_data.get('return_percentage', 'N/A')}%

Please provide insights on:
1. Investment quality based on Buffett principles
2. DCA performance evaluation
3. Risk assessment
4. Investment recommendations

Keep your response concise but comprehensive, focusing on actionable insights.
"""
        return prompt
    
    def query_ai(self, 
                 user_query: str, 
                 context_data: Optional[Dict[str, Any]] = None,
                 conversation_history: Optional[List[Dict[str, str]]] = None) -> str:
        """Send query to AI and get response."""
        if not self.is_ready():
            return "❌ AI is not configured. Please set up your Google AI API key."
        
        try:
            # Build the full prompt with context
            full_prompt = self._build_full_prompt(user_query, context_data, conversation_history)
            
            # Generate response
            response = self.model.generate_content(full_prompt)
            return response.text if response.text else "Sorry, I couldn't generate a response."
            
        except Exception as e:
            return f"❌ Error generating AI response: {str(e)}"
    
    def _build_full_prompt(self, 
                          user_query: str, 
                          context_data: Optional[Dict[str, Any]] = None,
                          conversation_history: Optional[List[Dict[str, str]]] = None) -> str:
        """Build a comprehensive prompt with context and history."""
        
        # Base system prompt
        system_prompt = """
You are an AI assistant specialized in financial analysis and Warren Buffett's investment strategies. 
You help users analyze stocks using the Buffett 11 Checklist and DCA (Dollar Cost Averaging) strategies.

Guidelines:
- Provide clear, actionable financial advice
- Focus on long-term value investing principles
- Explain complex concepts in simple terms
- Always consider risk management
- Be honest about limitations and uncertainties
- Use data provided in context when available
"""
        
        # Add conversation history if available
        history_text = ""
        if conversation_history:
            history_text = "\n\nPrevious conversation:\n"
            for msg in conversation_history[-3:]:  # Last 3 messages for context
                history_text += f"User: {msg.get('user', '')}\nAssistant: {msg.get('assistant', '')}\n"
        
        # Add context data if available
        context_text = ""
        if context_data:
            context_text = f"\n\nCurrent analysis context:\n{self._format_context_data(context_data)}"
        
        # Combine all parts
        full_prompt = f"{system_prompt}{history_text}{context_text}\n\nUser Query: {user_query}\n\nPlease provide a helpful response:"
        
        return full_prompt
    
    def _format_context_data(self, context_data: Dict[str, Any]) -> str:
        """Format context data for the prompt."""
        formatted = ""
        
        if 'selected_stocks' in context_data:
            formatted += f"Selected stocks: {', '.join(context_data['selected_stocks'])}\n"
        
        if 'market' in context_data:
            formatted += f"Market: {context_data['market']}\n"
        
        if 'analysis_results' in context_data:
            formatted += "Analysis results available for detailed questions.\n"
        
        if 'dca_settings' in context_data:
            settings = context_data['dca_settings']
            formatted += f"DCA settings: {settings.get('monthly_invest', 'N/A')} per month, {settings.get('period', 'N/A')} period\n"
        
        return formatted
    
    def get_sample_questions(self) -> List[str]:
        """Get sample questions users can ask."""
        return [
            "Analyze the selected stocks based on Warren Buffett's principles",
            "What are the key risks in my stock selection?",
            "How does the DCA simulation look for my portfolio?",
            "Which stock shows the best Buffett checklist score?",
            "Should I increase my monthly DCA investment?",
            "What market trends should I be aware of?",
            "Compare the dividend yields of my selected stocks",
            "How do these stocks perform during economic downturns?",
            "What's the ideal portfolio allocation for these stocks?",
            "When should I consider selling these positions?"
        ]