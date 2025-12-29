import streamlit as st
import yfinance as yf
import pandas as pd
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from ta.momentum import RSIIndicator
from ta.trend import MACD

st.set_page_config(page_title="Gen-AI Stock Analyst", layout="wide")

st.title("📈 Gen-AI Fundamental + Technical Stock Analyst (No API Key Needed)")

stock = st.text_input("Enter Stock Ticker (Example: RELIANCE.NS / AAPL)")

if st.button("Analyze Stock"):

    if stock == "":
        st.warning("Please enter a stock ticker")
    else:
        ticker = yf.Ticker(stock)

        # ================= FUNDAMENTALS =================
        info = ticker.info
        name = info.get("longName")
        sector = info.get("sector")
        industry = info.get("industry")
        market_cap = info.get("marketCap")
        rev = info.get("totalRevenue")
        profit = info.get("grossProfits")
        pe = info.get("trailingPE")
        pb = info.get("priceToBook")
        roe = info.get("returnOnEquity")
        debt = info.get("debtToEquity")

        st.subheader("🏢 Company & Financial Summary")
        st.write(f"""
        **Name:** {name}  
        **Sector:** {sector}  
        **Industry:** {industry}  
        **Market Cap:** {market_cap}  
        **Revenue:** {rev}  
        **Profit:** {profit}  
        **PE Ratio:** {pe}  
        **PB Ratio:** {pb}  
        **ROE:** {roe}  
        **Debt to Equity:** {debt}  
        """)

        # ================= NEWS =================
    st.subheader("📰 Latest News Headlines")

    news_text = ""
        try:
    news = ticker.news
        if news and len(news) > 0:
        for n in news[:5]:
            title = n.get("title", "No Title Available")
            source = n.get("publisher", "Unknown Source")
            st.write(f"- **{title}** ({source})")
            news_text += title + "\n"
        else:
        st.info("No recent news available for this stock.")
        news_text = "No major recent news available."
        except:
        st.info("News data could not be fetched.")
        news_text = "No major recent news available."

        # ================= TECHNICALS =================
    st.subheader("📊 Technical Indicators")

        data = ticker.history(period="1y")
        close = data["Close"]

        # SMA
        data["SMA_50"] = close.rolling(50).mean()
        data["SMA_200"] = close.rolling(200).mean()

        if data["SMA_50"].iloc[-1] > data["SMA_200"].iloc[-1]:
            sma_signal = "Bullish (Golden Cross Trend)"
        else:
            sma_signal = "Bearish (Below Long Term Trend)"

        # RSI
        rsi = RSIIndicator(close, window=14).rsi().iloc[-1]
        if rsi > 70:
            rsi_signal = "Overbought – Possible Correction"
        elif rsi < 30:
            rsi_signal = "Oversold – Possible Reversal"
        else:
            rsi_signal = "Neutral Momentum"

        # MACD
        macd = MACD(close)
        macd_value = macd.macd().iloc[-1]
        signal_value = macd.macd_signal().iloc[-1]

        if macd_value > signal_value:
            macd_signal = "Bullish Momentum"
        else:
            macd_signal = "Bearish Momentum"

        st.write(f"**SMA Trend:** {sma_signal}")
        st.write(f"**RSI:** {rsi:.2f} → {rsi_signal}")
        st.write(f"**MACD:** {macd_signal}")

        # ===================== AI MODEL =====================
        st.subheader("🤖 Gen-AI Analysis")

        model_name = "mistralai/Mistral-7B-Instruct-v0.2"
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForCausalLM.from_pretrained(model_name)
        llm = pipeline("text-generation", model=model, tokenizer=tokenizer)

        prompt = f"""
        You are a professional equity research analyst.
        Analyze this stock fundamentally and technically and give a final investment view.

        Company: {name}
        Sector: {sector}
        Industry: {industry}
        Market Cap: {market_cap}
        Revenue: {rev}
        Profit: {profit}
        PE: {pe}
        PB: {pb}
        ROE: {roe}
        Debt to Equity: {debt}

        News Headlines:
        {news_text}

        Technicals:
        SMA Trend: {sma_signal}
        RSI: {rsi_signal} ({rsi})
        MACD: {macd_signal}

        Provide output in this structure:
        1. Company Overview
        2. Financial Health
        3. Growth Outlook
        4. Key Risks
        5. News Sentiment
        6. Technical Trend Summary
        7. Final Verdict: Bullish / Bearish / Neutral
        """

        response = llm(prompt, max_length=900)
        st.success(response[0]["generated_text"])


