"""
AI DCA Helper Module
Provides AI analysis for DCA optimization results with provider abstraction
"""

import json
from datetime import datetime
from typing import Dict, List, Optional, Any
from abc import ABC, abstractmethod


class AIProvider(ABC):
    """Abstract base class for AI providers"""
    
    @abstractmethod
    def generate_analysis(self, prompt: str, max_tokens: int = 1000) -> str:
        """Generate AI analysis from prompt"""
        pass
    
    @abstractmethod
    def is_configured(self) -> bool:
        """Check if the provider is properly configured"""
        pass


class GeminiProvider(AIProvider):
    """Google Gemini AI provider"""
    
    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key
        self._client = None
        
        if api_key:
            try:
                import google.generativeai as genai
                genai.configure(api_key=api_key)
                self._client = genai.GenerativeModel('gemini-pro')
            except ImportError:
                print("Warning: google-generativeai package not installed")
            except Exception as e:
                print(f"Warning: Failed to initialize Gemini: {e}")
    
    def generate_analysis(self, prompt: str, max_tokens: int = 1000) -> str:
        """Generate analysis using Gemini"""
        if not self._client:
            return "❌ Gemini API ไม่พร้อมใช้งาน (กรุณาตรวจสอบ API key)"
        
        try:
            response = self._client.generate_content(prompt)
            return response.text
        except Exception as e:
            return f"❌ เกิดข้อผิดพลาดใน Gemini API: {str(e)}"
    
    def is_configured(self) -> bool:
        """Check if Gemini is properly configured"""
        return self._client is not None


class OpenAIProvider(AIProvider):
    """OpenAI GPT provider"""
    
    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key
        self._client = None
        
        if api_key:
            try:
                import openai
                self._client = openai.OpenAI(api_key=api_key)
            except ImportError:
                print("Warning: openai package not installed")
            except Exception as e:
                print(f"Warning: Failed to initialize OpenAI: {e}")
    
    def generate_analysis(self, prompt: str, max_tokens: int = 1000) -> str:
        """Generate analysis using OpenAI GPT"""
        if not self._client:
            return "❌ OpenAI API ไม่พร้อมใช้งาน (กรุณาตรวจสอบ API key)"
        
        try:
            response = self._client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=max_tokens,
                temperature=0.7
            )
            return response.choices[0].message.content
        except Exception as e:
            return f"❌ เกิดข้อผิดพลาดใน OpenAI API: {str(e)}"
    
    def is_configured(self) -> bool:
        """Check if OpenAI is properly configured"""
        return self._client is not None


class DCAAnalysisHelper:
    """Main DCA AI analysis helper with provider abstraction"""
    
    def __init__(self):
        self.providers = {}
    
    def add_provider(self, name: str, provider: AIProvider):
        """Add an AI provider"""
        self.providers[name] = provider
    
    def setup_gemini(self, api_key: str):
        """Setup Gemini provider"""
        provider = GeminiProvider(api_key)
        self.add_provider('gemini', provider)
        return provider.is_configured()
    
    def setup_openai(self, api_key: str):
        """Setup OpenAI provider"""
        provider = OpenAIProvider(api_key)
        self.add_provider('openai', provider)
        return provider.is_configured()
    
    def get_available_providers(self) -> List[str]:
        """Get list of configured providers"""
        return [name for name, provider in self.providers.items() if provider.is_configured()]
    
    def _compress_optimization_results(self, optimization_results: Dict, max_results: int = 5) -> Dict:
        """Compress optimization results for efficient AI processing"""
        compressed = {
            'ticker': optimization_results.get('ticker', ''),
            'strategy': optimization_results.get('strategy', ''),
            'total_combinations_tested': optimization_results.get('total_combinations_tested', 0),
            'ranking_criteria': optimization_results.get('ranking_criteria', ''),
            'top_results': []
        }
        
        # Get top N results
        all_results = optimization_results.get('all_results', [])
        top_results = all_results[:max_results]
        
        for result in top_results:
            compressed_result = {
                'rank': result.get('rank', 0),
                'parameters': result.get('parameters', {}),
                'total_return_pct': result.get('total_return_pct', 0),
                'cost_basis': result.get('cost_basis', 0),
                'total_invested': result.get('total_invested', 0),
                'max_drawdown_pct': result.get('max_drawdown_pct', 0),
                'sharpe_ratio': result.get('sharpe_ratio', 0),
                'break_even_achieved': result.get('break_even_achieved', False),
                'break_even_forecast': result.get('break_even_forecast', {}),
                'time_in_profit_pct': result.get('time_in_profit_pct', 0)
            }
            compressed['top_results'].append(compressed_result)
        
        return compressed
    
    def _create_single_ticker_prompt(self, compressed_results: Dict) -> str:
        """Create analysis prompt for single ticker results"""
        
        ticker = compressed_results['ticker']
        strategy = compressed_results['strategy']
        top_result = compressed_results['top_results'][0] if compressed_results['top_results'] else {}
        
        prompt = f"""
คุณเป็นนักวิเคราะห์การลงทุนที่เชี่ยวชาญ DCA (Dollar Cost Averaging) กรุณาวิเคราะห์ผลการเพิ่มประสิทธิภาพ DCA สำหรับหุ้น {ticker} ด้วยกลยุทธ์ {strategy}

ข้อมูลการเพิ่มประสิทธิภาพ:
- จำนวนชุดพารามิเตอร์ที่ทดสอบ: {compressed_results.get('total_combinations_tested', 0)}
- เกณฑ์การจัดอันดับ: {compressed_results.get('ranking_criteria', '')}

ผลลัพธ์ 5 อันดับแรก:
"""
        
        for i, result in enumerate(compressed_results['top_results'][:5]):
            prompt += f"""
อันดับ {i+1}:
- พารามิเตอร์: {json.dumps(result['parameters'], ensure_ascii=False)}
- ผลตอบแทนรวม: {result['total_return_pct']:.2f}%
- ราคาเฉลี่ยที่ซื้อ: {result['cost_basis']:.2f}
- เงินลงทุนรวม: {result['total_invested']:,.0f}
- Drawdown สูงสุด: {result['max_drawdown_pct']:.2f}%
- Sharpe Ratio: {result['sharpe_ratio']:.3f}
- คุ้มทุนแล้ว: {'ใช่' if result['break_even_achieved'] else 'ไม่'}
- เวลาที่มีกำไร: {result['time_in_profit_pct']:.1f}%
"""

        prompt += """

กรุณาวิเคราะห์และสรุปเป็นภาษาไทยในรูปแบบ bullet points:

🎯 **กลยุทธ์ที่แนะนำ:**
- [อธิบายพารามิเตอร์ที่ดีที่สุดและเหตุผล]

⚡ **จุดแข็ง:**
- [จุดเด่นของการตั้งค่าที่ดีที่สุด]

⚠️ **ข้อควรระวัง:**
- [ความเสี่ยงและข้อจำกัด รวมทั้งเตือนเรื่อง overfitting หากพารามิเตอร์ซับซ้อนแต่ผลดีกว่าเพียงเล็กน้อย]

💡 **คำแนะนำปรับปรุง:**
- [แนะนำการปรับแต่งเพิ่มเติม]

📊 **คาดการณ์จุดคุ้มทุน:**
- [วิเคราะห์สถานการณ์คุ้มทุนและแนวโน้ม]
"""
        
        return prompt
    
    def _create_multi_ticker_prompt(self, multi_ticker_results: Dict) -> str:
        """Create analysis prompt for multi-ticker results"""
        
        prompt = """
คุณเป็นนักวิเคราะห์การลงทุนที่เชี่ยวชาญ DCA กรุณาวิเคราะห์ผลการเพิ่มประสิทธิภาพ DCA สำหรับหลายหุ้น

สรุปผลลัพธ์แต่ละหุ้น:
"""
        
        summary_by_ticker = multi_ticker_results.get('summary_by_ticker', {})
        
        for ticker, summary in summary_by_ticker.items():
            if summary:
                prompt += f"""
{ticker}:
- กลยุทธ์ที่ดีที่สุด: {summary.get('strategy', 'N/A')}
- ผลตอบแทน: {summary.get('metrics', {}).get('total_return_pct', 0):.2f}%
- ราคาเฉลี่ยที่ซื้อ: {summary.get('metrics', {}).get('cost_basis', 0):.2f}
- Sharpe Ratio: {summary.get('metrics', {}).get('sharpe_ratio', 0):.3f}
- คุ้มทุนแล้ว: {'ใช่' if summary.get('metrics', {}).get('break_even_achieved', False) else 'ไม่'}
"""
        
        prompt += """

กรุณาวิเคราะห์และสรุปเป็นภาษาไทยในรูปแบบ bullet points:

🏆 **อันดับหุ้นที่แนะนำ:**
- [จัดอันดับหุ้นตามประสิทธิภาพและความเสี่ยง]

💰 **กลยุทธ์ที่โดดเด่น:**
- [กลยุทธ์ไหนที่ใช้ได้ผลดีกับหลายหุ้น]

⚖️ **การกระจายความเสี่ยง:**
- [แนะนำสัดส่วนการลงทุนในแต่ละหุ้น]

🎯 **ข้อเสนะแนะรวม:**
- หุ้นไหนควรเพิ่ม DCA
- หุ้นไหนควรลด DCA  
- หุ้นไหนควรหยุด DCA ชั่วคราว

⚠️ **ข้อควรระวัง:**
- [ความเสี่ยงโดยรวมและข้อแม้]

📈 **แนวโน้มระยะยาว:**
- [คาดการณ์และกลยุทธ์ระยะยาว]
"""
        
        return prompt
    
    def analyze_single_ticker(self, optimization_results: Dict, 
                            provider_name: str = 'auto') -> str:
        """
        Generate AI analysis for single ticker optimization results
        
        Args:
            optimization_results: Results from DCAOptimizer
            provider_name: AI provider to use ('auto', 'gemini', 'openai')
            
        Returns:
            AI analysis in Thai
        """
        if not self.providers:
            return "❌ ไม่มี AI provider ที่พร้อมใช้งาน กรุณาใส่ API key"
        
        # Select provider
        if provider_name == 'auto':
            available = self.get_available_providers()
            if not available:
                return "❌ ไม่มี AI provider ที่พร้อมใช้งาน"
            provider_name = available[0]  # Use first available
        
        if provider_name not in self.providers:
            return f"❌ ไม่พบ provider '{provider_name}'"
        
        provider = self.providers[provider_name]
        if not provider.is_configured():
            return f"❌ Provider '{provider_name}' ไม่พร้อมใช้งาน"
        
        try:
            # Compress results for AI processing
            compressed = self._compress_optimization_results(optimization_results)
            
            # Create prompt
            prompt = self._create_single_ticker_prompt(compressed)
            
            # Generate analysis
            analysis = provider.generate_analysis(prompt, max_tokens=1500)
            
            # Add metadata
            metadata = f"""
---
📋 **ข้อมูลการวิเคราะห์**
- วิเคราะห์โดย: {provider_name.title()}
- วันที่: {datetime.now().strftime('%Y-%m-%d %H:%M')}
- หุ้น: {compressed['ticker']}
- กลยุทธ์: {compressed['strategy']}
- ชุดพารามิเตอร์ที่ทดสอบ: {compressed['total_combinations_tested']}

---
"""
            
            return metadata + analysis
            
        except Exception as e:
            return f"❌ เกิดข้อผิดพลาดในการวิเคราะห์: {str(e)}"
    
    def analyze_multi_ticker(self, multi_ticker_results: Dict, 
                           provider_name: str = 'auto') -> str:
        """
        Generate AI analysis for multi-ticker optimization results
        
        Args:
            multi_ticker_results: Results from multi-ticker optimization
            provider_name: AI provider to use
            
        Returns:
            AI analysis in Thai
        """
        if not self.providers:
            return "❌ ไม่มี AI provider ที่พร้อมใช้งาน กรุณาใส่ API key"
        
        # Select provider
        if provider_name == 'auto':
            available = self.get_available_providers()
            if not available:
                return "❌ ไม่มี AI provider ที่พร้อมใช้งาน"
            provider_name = available[0]
        
        if provider_name not in self.providers:
            return f"❌ ไม่พบ provider '{provider_name}'"
        
        provider = self.providers[provider_name]
        if not provider.is_configured():
            return f"❌ Provider '{provider_name}' ไม่พร้อมใช้งาน"
        
        try:
            # Create prompt
            prompt = self._create_multi_ticker_prompt(multi_ticker_results)
            
            # Generate analysis
            analysis = provider.generate_analysis(prompt, max_tokens=2000)
            
            # Add metadata
            tickers = list(multi_ticker_results.get('summary_by_ticker', {}).keys())
            
            metadata = f"""
---
📋 **ข้อมูลการวิเคราะห์รวม**
- วิเคราะห์โดย: {provider_name.title()}
- วันที่: {datetime.now().strftime('%Y-%m-%d %H:%M')}
- หุ้นที่วิเคราะห์: {', '.join(tickers)}
- จำนวนหุ้น: {len(tickers)}

---
"""
            
            return metadata + analysis
            
        except Exception as e:
            return f"❌ เกิดข้อผิดพลาดในการวิเคราะห์: {str(e)}"
    
    def analyze_break_even_forecast(self, metrics: Dict, 
                                  monte_carlo_results: Optional[Dict] = None,
                                  provider_name: str = 'auto') -> str:
        """
        Generate detailed break-even analysis
        
        Args:
            metrics: DCA metrics including break-even info
            monte_carlo_results: Optional Monte Carlo simulation results
            provider_name: AI provider to use
            
        Returns:
            Break-even analysis in Thai
        """
        if not self.providers:
            return "❌ ไม่มี AI provider ที่พร้อมใช้งาน"
        
        # Select provider
        if provider_name == 'auto':
            available = self.get_available_providers()
            if not available:
                return "❌ ไม่มี AI provider ที่พร้อมใช้งาน"
            provider_name = available[0]
        
        provider = self.providers[provider_name]
        if not provider.is_configured():
            return f"❌ Provider '{provider_name}' ไม่พร้อมใช้งาน"
        
        try:
            # Create detailed break-even prompt
            prompt = f"""
คุณเป็นนักวิเคราะห์การลงทุนที่เชี่ยวชาญการคาดการณ์จุดคุ้มทุน (Break-even) สำหรับ DCA

ข้อมูลปัจจุบัน:
- หุ้น: {metrics.get('ticker', 'N/A')}
- ราคาเฉลี่ยที่ซื้อ: {metrics.get('cost_basis', 0):.2f}
- ราคาปัจจุบัน: {metrics.get('final_price', 0):.2f}
- คุ้มทุนแล้ว: {'ใช่' if metrics.get('break_even_achieved', False) else 'ไม่'}

การคาดการณ์จุดคุ้มทุน:
{json.dumps(metrics.get('break_even_forecast', {}), ensure_ascii=False, indent=2)}
"""

            if monte_carlo_results:
                prompt += f"""

ผล Monte Carlo Simulation:
- ความน่าจะเป็นที่จะคุ้มทุน: {monte_carlo_results.get('break_even_probability', 0)*100:.1f}%
- วันที่คาดว่าจะคุ้มทุน (Median): {monte_carlo_results.get('median_days_to_breakeven', 'N/A')} วัน
- วันที่คาดว่าจะคุ้มทุน (75th percentile): {monte_carlo_results.get('percentile_75_days', 'N/A')} วัน
"""

            prompt += """

กรุณาวิเคราะห์และสรุปเป็นภาษาไทย:

🎯 **สถานการณ์จุดคุ้มทุน:**
- [อธิบายสถานการณ์ปัจจุบัน]

📊 **การคาดการณ์:**
- [วิเคราะห์ความน่าจะเป็นและกรอบเวลา]

⚠️ **ปัจจัยความเสี่ยง:**
- [ปัจจัยที่อาจทำให้คาดการณ์เปลี่ยนแปลง]

💡 **กลยุทธ์แนะนำ:**
- [แนะนำการปรับแต่งเพื่อเร่งจุดคุ้มทุน]

🔮 **ความน่าเชื่อถือ:**
- [ประเมินความน่าเชื่อถือของการคาดการณ์]
"""
            
            analysis = provider.generate_analysis(prompt, max_tokens=1200)
            
            return f"""
---
📋 **การวิเคราะห์จุดคุ้มทุน (Break-even Analysis)**
- วิเคราะห์โดย: {provider_name.title()}
- วันที่: {datetime.now().strftime('%Y-%m-%d %H:%M')}

---
{analysis}
"""
            
        except Exception as e:
            return f"❌ เกิดข้อผิดพลาดในการวิเคราะห์จุดคุ้มทุน: {str(e)}"
    
    def create_summary_report(self, optimization_summary: Dict, 
                            provider_name: str = 'auto') -> str:
        """
        Create comprehensive summary report
        
        Args:
            optimization_summary: Complete optimization results
            provider_name: AI provider to use
            
        Returns:
            Comprehensive summary report in Thai
        """
        if not self.providers:
            return "❌ ไม่มี AI provider ที่พร้อมใช้งาน"
        
        # Combine individual analyses
        multi_ticker_analysis = self.analyze_multi_ticker(optimization_summary, provider_name)
        
        # Add executive summary
        try:
            tickers = optimization_summary.get('tickers_optimized', [])
            strategies = optimization_summary.get('strategies_tested', [])
            
            executive_summary = f"""
# 📊 สรุปผลการเพิ่มประสิทธิภาพ DCA

## 🎯 ภาพรวมการวิเคราะห์
- **หุ้นที่วิเคราะห์:** {len(tickers)} ตัว ({', '.join(tickers)})
- **กลยุทธ์ที่ทดสอบ:** {len(strategies)} แบบ
- **วันที่วิเคราะห์:** {datetime.now().strftime('%Y-%m-%d %H:%M')}

---

{multi_ticker_analysis}

---

## 📁 ข้อมูลเพิ่มเติม
- ผลลัพธ์รายละเอียดสามารถดาวน์โหลดในไฟล์ Excel
- การวิเคราะห์ใช้ข้อมูลประวัติการซื้อขายและไม่รับประกันผลลัพธ์ในอนาคต
- ควรทบทวนและปรับปรุงพารามิเตอร์เป็นระยะ

**หมายเหตุ:** การลงทุนมีความเสี่ยง ผู้ลงทุนควรศึกษาข้อมูลก่อนตัดสินใจลงทุน
"""
            
            return executive_summary
            
        except Exception as e:
            return f"❌ เกิดข้อผิดพลาดในการสร้างรายงานสรุป: {str(e)}"