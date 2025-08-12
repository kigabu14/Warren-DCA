# Warren-DCA: ระบบวิเคราะห์หุ้นและ Backtesting แบบครบวงจร (v0.2.0)

## 📋 ภาพรวมระบบ

**Warren-DCA** เป็นระบบวิเคราะห์หุ้นและการลงทุนแบบครบวงจร ที่รวมแนวทางการลงทุนของ Warren Buffett เข้ากับเทคโนลยี Backtesting สมัยใหม่

### ✨ คุณสมบัติหลัก

🔍 **วิเคราะห์หุ้น**
- การตรวจสอบตาม Buffett 18 Checklist 
- จำลองการลงทุนแบบ DCA (Dollar Cost Averaging)
- คำนวณผลตอบแทนเงินปันผล
- รองรับหุ้นจากตลาดทั่วโลก (US, SET100, Europe, Asia, Australia)

🚀 **ระบบ Backtesting**
- 4 กลยุทธ์การลงทุน: Bollinger Bands, MA Crossover, RSI, Buy & Hold
- คำนวณตัวชี้วัดครบถ้วน: CAGR, Sharpe Ratio, Max Drawdown, Win Rate
- รองรับค่าธรรมเนียมและ Risk Management
- ส่งออกผลลัพธ์เป็นไฟล์ Excel

📚 **เอกสารภาษาไทย**
- คู่มือการใช้งานแบบละเอียด
- คำอธิบายกลยุทธ์และตัวชี้วัด
- คำเตือนความเสี่ยงและข้อจำกัด

## 🛠️ การติดตั้งและใช้งาน

### ความต้องการของระบบ
```
Python 3.7+
streamlit>=1.28.0
yfinance
pandas
openpyxl
xlsxwriter
matplotlib
numpy
```

### การติดตั้ง
```bash
# โคลนโปรเจค
git clone https://github.com/kigabu14/Warren-DCA.git
cd Warren-DCA

# ติดตั้ง dependencies
pip install -r requirements.txt

# รันระบบ
streamlit run streamlit_app.py
```

## 📊 กลยุทธ์การลงทุน

### 1. 📈 Bollinger Bands Strategy
**หลักการ:**
- ซื้อเมื่อราคาแตะแถบล่าง (Lower Band)
- ขายเมื่อราคาแตะแถบบน (Upper Band)
- มี Stop Loss และ Take Profit เพิ่มเติม

**พารามิเตอร์:**
- **Period**: 20 วัน (ค่าเริ่มต้น)
- **Multiplier**: 2.0 (ค่าเริ่มต้น)
- **Stop Loss**: 5% (ค่าเริ่มต้น)
- **Take Profit**: 10% (ค่าเริ่มต้น)

### 2. ↗️ MA Crossover Strategy
**หลักการ:**
- ซื้อเมื่อ Moving Average สั้นตัดขึ้นเหนือ MA ยาว
- ขายเมื่อ Moving Average สั้นตัดลงต่ำกว่า MA ยาว
- มี Stop Loss และ Take Profit เพิ่มเติม

**พารามิเตอร์:**
- **Short Period**: 10 วัน (ค่าเริ่มต้น)
- **Long Period**: 50 วัน (ค่าเริ่มต้น)
- **Stop Loss**: 5% (ค่าเริ่มต้น)
- **Take Profit**: 10% (ค่าเริ่มต้น)

### 3. 📊 RSI Strategy
**หลักการ:**
- ซื้อเมื่อ RSI ต่ำกว่าระดับ Oversold
- ขายเมื่อ RSI สูงกว่าระดับ Overbought
- มี Stop Loss และ Take Profit เพิ่มเติม

**พารามิเตอร์:**
- **RSI Period**: 14 วัน (ค่าเริ่มต้น)
- **Oversold Level**: 30 (ค่าเริ่มต้น)
- **Overbought Level**: 70 (ค่าเริ่มต้น)
- **Stop Loss**: 5% (ค่าเริ่มต้น)
- **Take Profit**: 10% (ค่าเริ่มต้น)

### 4. 🏦 Buy & Hold Strategy
**หลักการ:**
- ซื้อหุ้นในวันแรกของช่วงทดสอบ
- ถือหุ้นตลอดช่วงเวลา
- ขายหุ้นในวันสุดท้าย

**หมายเหตุ:** Stop Loss และ Take Profit ไม่ใช้งานในกลยุทธ์นี้

## 📈 ตัวชี้วัดผลการดำเนินงาน

### 💰 ตัวชี้วัดผลตอบแทน
- **Final Value**: มูลค่าสุดท้ายของพอร์ต
- **Net Return (%)**: ผลตอบแทนสุทธิ
- **CAGR (%)**: อัตราผลตอบแทนแบบทบต้นต่อปี

### 📉 ตัวชี้วัดความเสี่ยง
- **Max Drawdown (%)**: การลดลงสูงสุดจากจุดสูงสุด
- **Sharpe Ratio**: อัตราส่วนผลตอบแทนต่อความเสี่ยง

### 🔄 ตัวชี้วัดการซื้อขาย
- **Number of Trades**: จำนวนรอบการซื้อขาย
- **Win Rate (%)**: อัตราการทำกำไร
- **Profit Factor**: อัตราส่วนกำไรรวมต่อขาดทุนรวม
- **Average Profit per Trade (%)**: กำไรเฉลี่ยต่อเทรด

## 🎯 วิธีการใช้งาน

### 📊 การวิเคราะห์หุ้น
1. เลือกตลาดหุ้น (US, SET100, Europe, Asia, Australia, Global)
2. เลือกหุ้นที่ต้องการวิเคราะห์
3. ตั้งค่าช่วงเวลาและจำนวนเงินลงทุน DCA
4. คลิก "วิเคราะห์"
5. ดูผลลัพธ์และดาวน์โหลด Excel

### 🚀 การ Backtesting
1. เลือกกลยุทธ์ที่ต้องการทดสอบ
2. เลือกตลาดหุ้นและหุ้นที่ต้องการ
3. ตั้งค่าเงินทุนเริ่มต้นและค่าธรรมเนียม
4. ปรับพารามิเตอร์ของกลยุทธ์
5. คลิก "เริ่มทดสอบ"
6. ดูผลลัพธ์และดาวน์โหลด Excel

## 📋 Buffett 18 Checklist

### กฎการลงทุนตาม Warren Buffett
1. Inventory & Net Earnings เพิ่มขึ้นต่อเนื่อง
2. ไม่มี R&D
3. EBITDA > Current Liabilities ทุกปี
4. PPE เพิ่มขึ้น (ไม่มี spike)
5. RTA ≥ 11%
6. RTA ≥ 17%
7. LTD/Total Assets ≤ 0.5
8. EBITDA ปีล่าสุดจ่ายหนี้ LTD หมดใน ≤ 4 ปี
9. Equity ติดลบในปีใดหรือไม่
10. DSER ≤ 1.0
11. DSER ≤ 0.8
12. ไม่มี Preferred Stock
13. Retained Earnings เติบโต ≥ 7%
14. Retained Earnings เติบโต ≥ 13.5%
15. Retained Earnings เติบโต ≥ 17%
16. มี Treasury Stock
17. ROE ≥ 23%
18. Goodwill เพิ่มขึ้น

## ⚠️ ข้อจำกัดและสมมติฐาน

### ข้อจำกัดของระบบ
- ไม่มีการคำนวณ Slippage
- ใช้ข้อมูลย้อนหลังเท่านั้น
- มี Survivorship Bias
- ไม่รวมเงินปันผลใน Backtesting
- ใช้ราคาปิดในการซื้อขาย

### สมมติฐาน
- สามารถซื้อขายได้ในปริมาณที่ต้องการ
- ค่าธรรมเนียมคงที่ทุกครั้ง
- ไม่มีต้นทุนการถือครอง
- ข้อมูลราคาถูกต้องและครบถ้วน

## 🚨 คำเตือนความเสี่ยง

⚠️ **คำเตือนสำคัญ:**
- ผลการดำเนินงานในอดีตไม่ใช่การรับประกันผลตอบแทนในอนาคต
- การลงทุนมีความเสี่ยง อาจได้รับกำไรหรือขาดทุนได้
- ควรศึกษาข้อมูลและปรึกษาผู้เชี่ยวชาญก่อนตัดสินใจลงทุน
- ระบบนี้เป็นเครื่องมือช่วยวิเคราะห์เท่านั้น ไม่ใช่คำแนะนำการลงทุน
- ผู้ใช้ต้องรับผิดชอบการตัดสินใจลงทุนด้วยตนเอง

## 📧 การติดต่อ

- **GitHub**: [kigabu14/Warren-DCA](https://github.com/kigabu14/Warren-DCA)
- **Issues**: [GitHub Issues](https://github.com/kigabu14/Warren-DCA/issues)

## 📄 สัญญาอนุญาต

โปรเจคนี้อยู่ภายใต้สัญญาอนุญาต Apache License 2.0

---

**Warren-DCA v0.2.0** | Powered by Yahoo Finance | สร้างด้วย ❤️ สำหรับนักลงทุนไทย