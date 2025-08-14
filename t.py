          st.subheader("🤖 การวิเคราะห์ AI")
                        
                        # Get Buffett score for context
                        fin = stock.financials
                        bs = stock.balance_sheet
                        cf = stock.cashflow
                        div = stock.dividends
                        hist_full = stock.history(period="1y")
                        
                        buffett_detail = buffett_11_checks_detail(fin, bs, cf, div, hist_full)
                        buffett_score = buffett_detail['score_pct']
                        
                        ai_prompt = f"""
                        วิเคราะห์หุ้น {selected_ticker} ในภาษาไทยอย่างสั้นกระชับ:
                        
                        ข้อมูลพื้นฐาน:
                        - คะแนน Buffett: {buffett_score}%
                        - การเปลี่ยนแปลงราคาวันนี้: {price_change:.2f}%
                        
                        ข้อมูล Sentiment จากข่าว:
                        - ข่าวบวก: {positive_count} ข่าว
                        - ข่าวลบ: {negative_count} ข่าว  
                        - ข่าวเป็นกลาง: {neutral_count} ข่าว
                        - Sentiment เฉลี่ย: {avg_sentiment:.3f}
                        
                        กรุณาให้การวิเคราะห์สั้นๆ ภายใน 3-4 ประโยค เกี่ยวกับโอกาสและความเสี่ยง
                        """
                        
                        with st.spinner("AI กำลังวิเคราะห์..."):
                            ai_analysis = get_ai_insights(gemini_api_key, ai_prompt)
                            st.write(ai_analysis)