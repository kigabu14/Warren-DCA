"""
Database features for Streamlit app.
Contains components for managing stored stock data and historical analysis.
"""

import streamlit as st
import pandas as pd
from datetime import datetime, timedelta
from typing import List, Dict, Any
import plotly.graph_objects as go
import plotly.express as px

from enhanced_data_loader import EnhancedDCADataLoader


class DatabaseFeatures:
    """Class to handle database-related features in Streamlit app."""
    
    def __init__(self):
        """Initialize database features."""
        # Use mock data mode in sandboxed environment
        self.enhanced_loader = EnhancedDCADataLoader(use_mock_data=True)
    
    def render_data_storage_section(self):
        """Render the data storage and historical analysis section."""
        st.header("📊 ระบบเก็บข้อมูลหุ้นและวิเคราะห์ย้อนหลัง")
        
        # Create tabs for different functionalities
        tab1, tab2, tab3, tab4 = st.tabs([
            "ดึงและเก็บข้อมูล", 
            "ข้อมูลที่เก็บไว้", 
            "วิเคราะห์ย้อนหลัง", 
            "จัดการข้อมูล"
        ])
        
        with tab1:
            self._render_data_fetching_section()
        
        with tab2:
            self._render_stored_data_section()
        
        with tab3:
            self._render_historical_analysis_section()
        
        with tab4:
            self._render_data_management_section()
    
    def _render_data_fetching_section(self):
        """Render data fetching and storage section."""
        st.subheader("ดึงข้อมูลหุ้นและบันทึกลงฐานข้อมูล")
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            ticker = st.text_input(
                "รหัสหุ้น", 
                placeholder="เช่น AAPL, MSFT, หรือ PTT.BK", 
                help="ใส่รหัสหุ้นที่ต้องการดึงข้อมูล"
            )
        
        with col2:
            period_options = {
                "1 เดือน": 1,
                "3 เดือน": 3,
                "6 เดือน": 6,
                "1 ปี": 12,
                "2 ปี": 24,
                "5 ปี": 60
            }
            
            selected_period = st.selectbox(
                "ช่วงเวลาที่ต้องการ",
                options=list(period_options.keys()),
                index=3,  # Default to 1 year
                help="เลือกช่วงเวลาย้อนหลังที่ต้องการดึงข้อมูล"
            )
        
        if st.button("🔄 ดึงและเก็บข้อมูล", type="primary"):
            if ticker:
                with st.spinner(f"กำลังดึงข้อมูล {ticker} ย้อนหลัง {selected_period}..."):
                    try:
                        months = period_options[selected_period]
                        data = self.enhanced_loader.fetch_historical_data_for_period(ticker.upper(), months)
                        
                        if data:
                            st.success(f"✅ ดึงและเก็บข้อมูล {ticker} สำเร็จ!")
                            
                            # Show preview of stored data
                            self._show_data_preview(ticker.upper(), data)
                        else:
                            st.error("❌ ไม่สามารถดึงข้อมูลได้ กรุณาตรวจสอบรหัสหุ้น")
                    
                    except Exception as e:
                        st.error(f"❌ เกิดข้อผิดพลาด: {str(e)}")
            else:
                st.warning("⚠️ กรุณาใส่รหัสหุ้น")
    
    def _render_stored_data_section(self):
        """Render stored data viewing section."""
        st.subheader("ข้อมูลหุ้นที่เก็บในฐานข้อมูล")
        
        # Get list of stored stocks
        stored_stocks = self.enhanced_loader.list_stored_stocks()
        
        if not stored_stocks:
            st.info("📋 ยังไม่มีข้อมูลหุ้นในฐานข้อมูล กรุณาดึงข้อมูลก่อน")
            return
        
        # Display stored stocks table
        df_stocks = pd.DataFrame(stored_stocks)
        df_stocks['last_updated'] = pd.to_datetime(df_stocks['last_updated']).dt.strftime('%Y-%m-%d %H:%M')
        
        st.dataframe(
            df_stocks,
            column_config={
                "ticker": st.column_config.TextColumn("รหัสหุ้น"),
                "company_name": st.column_config.TextColumn("ชื่อบริษัท"),
                "sector": st.column_config.TextColumn("ภาค"),
                "industry": st.column_config.TextColumn("อุตสาหกรรม"),
                "last_updated": st.column_config.TextColumn("อัปเดตล่าสุด")
            },
            use_container_width=True,
            hide_index=True
        )
        
        # Stock selection for detailed view
        stock_options = [f"{row['ticker']} - {row['company_name'] or 'N/A'}" for row in stored_stocks]
        selected_stock = st.selectbox(
            "เลือกหุ้นเพื่อดูรายละเอียด",
            options=stock_options,
            help="เลือกหุ้นที่ต้องการดูข้อมูลรายละเอียด"
        )
        
        if selected_stock:
            ticker = selected_stock.split(" - ")[0]
            self._show_detailed_stock_data(ticker)
    
    def _render_historical_analysis_section(self):
        """Render historical analysis section."""
        st.subheader("วิเคราะห์ข้อมูลย้อนหลังแบบออฟไลน์")
        
        # Get stored stocks for selection
        stored_stocks = self.enhanced_loader.list_stored_stocks()
        
        if not stored_stocks:
            st.info("📋 ไม่มีข้อมูลหุ้นในฐานข้อมูล กรุณาดึงข้อมูลก่อน")
            return
        
        # Stock selection
        ticker_options = [stock['ticker'] for stock in stored_stocks]
        selected_ticker = st.selectbox(
            "เลือกหุ้นสำหรับวิเคราะห์",
            options=ticker_options,
            help="เลือกหุ้นที่ต้องการวิเคราะห์จากข้อมูลที่เก็บไว้"
        )
        
        if selected_ticker:
            # Get available periods for the selected stock
            available_periods = self.enhanced_loader.get_available_periods(selected_ticker)
            
            if available_periods:
                analysis_period = st.selectbox(
                    "เลือกช่วงเวลาสำหรับวิเคราะห์",
                    options=available_periods,
                    help="เลือกช่วงเวลาที่ต้องการวิเคราะห์"
                )
                
                if st.button("📈 วิเคราะห์ข้อมูล"):
                    self._perform_offline_analysis(selected_ticker, analysis_period)
            else:
                st.warning("⚠️ ไม่มีข้อมูลประวัติสำหรับหุ้นนี้")
    
    def _render_data_management_section(self):
        """Render data management section."""
        st.subheader("จัดการข้อมูลในฐานข้อมูล")
        
        stored_stocks = self.enhanced_loader.list_stored_stocks()
        
        if not stored_stocks:
            st.info("📋 ไม่มีข้อมูลหุ้นในฐานข้อมูล")
            return
        
        # Delete stock data
        st.write("**ลบข้อมูลหุ้น**")
        ticker_to_delete = st.selectbox(
            "เลือกหุ้นที่ต้องการลบ",
            options=[stock['ticker'] for stock in stored_stocks],
            help="เลือกหุ้นที่ต้องการลบออกจากฐานข้อมูล"
        )
        
        if st.button("🗑️ ลบข้อมูล", type="secondary"):
            if st.session_state.get('confirm_delete'):
                success = self.enhanced_loader.delete_stored_stock(ticker_to_delete)
                if success:
                    st.success(f"✅ ลบข้อมูล {ticker_to_delete} สำเร็จ")
                    st.rerun()
                else:
                    st.error("❌ ไม่สามารถลบข้อมูลได้")
                st.session_state['confirm_delete'] = False
            else:
                st.session_state['confirm_delete'] = True
                st.warning(f"⚠️ คุณแน่ใจที่จะลบข้อมูล {ticker_to_delete}? กดปุ่มลบอีกครั้งเพื่อยืนยัน")
    
    def _show_data_preview(self, ticker: str, data: Dict[str, Any]):
        """Show preview of stored data."""
        st.write("**ตัวอย่างข้อมูลที่เก็บ:**")
        
        if 'historical_prices' in data and not data['historical_prices'].empty:
            hist_df = data['historical_prices'].tail(5)
            st.write("📊 ข้อมูลราคา (5 วันล่าสุด):")
            st.dataframe(hist_df, use_container_width=True)
        
        if 'dividends' in data and not data['dividends'].empty:
            div_df = data['dividends'].tail(3)
            st.write("💰 ข้อมูลเงินปันผล (3 ครั้งล่าสุด):")
            st.dataframe(div_df, use_container_width=True)
    
    def _show_detailed_stock_data(self, ticker: str):
        """Show detailed stock data from database."""
        stored_data = self.enhanced_loader.get_stored_stock_data(ticker)
        
        if not stored_data:
            st.warning(f"⚠️ ไม่พบข้อมูลรายละเอียดสำหรับ {ticker}")
            return
        
        st.write(f"**ข้อมูลรายละเอียด {ticker}**")
        
        # Current stock data
        if stored_data['stock_data']:
            stock_data = stored_data['stock_data']
            
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("ราคาปัจจุบัน", f"${stock_data.get('current_price', 'N/A'):.2f}" if stock_data.get('current_price') else "N/A")
                st.metric("P/E Ratio", f"{stock_data.get('pe_ratio', 'N/A'):.2f}" if stock_data.get('pe_ratio') else "N/A")
            
            with col2:
                st.metric("ราคาเปิด", f"${stock_data.get('open_price', 'N/A'):.2f}" if stock_data.get('open_price') else "N/A")
                st.metric("EPS", f"${stock_data.get('eps'):.2f}" if stock_data.get('eps') is not None else "N/A")
            
            with col3:
                st.metric("ราคาปิด", f"${stock_data.get('close_price', 'N/A'):.2f}" if stock_data.get('close_price') else "N/A")
                st.metric("ROE", f"{stock_data.get('roe', 'N/A'):.2%}" if stock_data.get('roe') else "N/A")
            
            with col4:
                st.metric("ปันผล", f"{stock_data.get('dividend_yield', 'N/A'):.2%}" if stock_data.get('dividend_yield') else "N/A")
                st.metric("Market Cap", f"${stock_data.get('market_cap', 'N/A'):,.0f}" if stock_data.get('market_cap') else "N/A")
        
        # Historical price chart
        if stored_data['historical_data'] is not None and not stored_data['historical_data'].empty:
            st.write("**📈 กราฟราคาหุ้น**")
            hist_data = stored_data['historical_data']
            
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=hist_data.index,
                y=hist_data['close_price'],
                mode='lines',
                name='Close Price',
                line=dict(color='blue')
            ))
            
            fig.update_layout(
                title=f"ราคาหุ้น {ticker}",
                xaxis_title="วันที่",
                yaxis_title="ราคา",
                height=400
            )
            
            st.plotly_chart(fig, use_container_width=True)
        
        # Financial statements summary
        if stored_data['financial_statements']:
            st.write("**📊 สรุปงบการเงิน**")
            
            financial_data = []
            for statement in stored_data['financial_statements']:
                financial_data.append({
                    'ประเภท': statement['statement_type'],
                    'วันที่': statement['period_ending'],
                    'รายได้รวม': f"${statement.get('total_revenue', 'N/A'):,.0f}" if statement.get('total_revenue') else "N/A",
                    'กำไรสุทธิ': f"${statement.get('net_income', 'N/A'):,.0f}" if statement.get('net_income') else "N/A",
                    'สินทรัพย์รวม': f"${statement.get('total_assets', 'N/A'):,.0f}" if statement.get('total_assets') else "N/A"
                })
            
            if financial_data:
                df_financial = pd.DataFrame(financial_data)
                st.dataframe(df_financial, use_container_width=True, hide_index=True)
    
    def _perform_offline_analysis(self, ticker: str, period: str):
        """Perform offline analysis using stored data."""
        stored_data = self.enhanced_loader.get_stored_stock_data(ticker)
        
        if not stored_data or stored_data['historical_data'] is None or stored_data['historical_data'].empty:
            st.error("❌ ไม่มีข้อมูลประวัติสำหรับการวิเคราะห์")
            return
        
        hist_data = stored_data['historical_data']
        
        # Calculate period range
        if period == "1 เดือน":
            days = 30
        elif period == "3 เดือน":
            days = 90
        elif period == "1 ปี":
            days = 365
        else:
            days = 30
        
        end_date = hist_data.index.max()
        start_date = end_date - timedelta(days=days)
        period_data = hist_data[hist_data.index >= start_date]
        
        if period_data.empty:
            st.warning("⚠️ ไม่มีข้อมูลในช่วงเวลาที่เลือก")
            return
        
        st.success(f"✅ วิเคราะห์ข้อมูล {ticker} ช่วง {period} (ออฟไลน์)")
        
        # Basic statistics
        col1, col2, col3, col4 = st.columns(4)
        
        start_price = period_data['close_price'].iloc[0]
        end_price = period_data['close_price'].iloc[-1]
        price_change = ((end_price - start_price) / start_price) * 100
        
        high_price = period_data['high_price'].max()
        low_price = period_data['low_price'].min()
        avg_volume = period_data['volume'].mean()
        
        with col1:
            st.metric("การเปลี่ยนแปลงราคา", f"{price_change:.2f}%")
        
        with col2:
            st.metric("ราคาสูงสุด", f"${high_price:.2f}")
        
        with col3:
            st.metric("ราคาต่ำสุด", f"${low_price:.2f}")
        
        with col4:
            st.metric("ปริมาณเฉลี่ย", f"{avg_volume:,.0f}")
        
        # Price chart for the period
        fig = go.Figure()
        
        fig.add_trace(go.Candlestick(
            x=period_data.index,
            open=period_data['open_price'],
            high=period_data['high_price'],
            low=period_data['low_price'],
            close=period_data['close_price'],
            name=ticker
        ))
        
        fig.update_layout(
            title=f"กราฟเทียน {ticker} - {period}",
            xaxis_title="วันที่",
            yaxis_title="ราคา",
            height=500
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Volume chart
        fig_vol = go.Figure()
        fig_vol.add_trace(go.Bar(
            x=period_data.index,
            y=period_data['volume'],
            name='Volume',
            marker_color='rgba(0,100,80,0.8)'
        ))
        
        fig_vol.update_layout(
            title=f"ปริมาณการซื้อขาย {ticker} - {period}",
            xaxis_title="วันที่",
            yaxis_title="ปริมาณ",
            height=300
        )
        
        st.plotly_chart(fig_vol, use_container_width=True)