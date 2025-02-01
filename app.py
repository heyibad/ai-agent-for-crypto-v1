import os
import time
from datetime import datetime, timedelta

import streamlit as st
import pandas as pd
import requests
import plotly.express as px
import plotly.graph_objects as go
from dotenv import load_dotenv

# Import CrewAI components 
from crewai import Task, Agent, Crew, LLM
from crewai_tools import SerperDevTool


# Environment Setup & Initialization

load_dotenv()
COINMARKETCAP_API_KEY= os.getenv('COINMARKETCAP_API_KEY')
GEMINI_API_KEY = os.getenv('GEMINI_API_KEY')
MODEL = os.getenv('MODEL')

if not COINMARKETCAP_API_KEY:
    raise ValueError("Missing environment variable: COINMARKETCAP_API_KEY")
if not GEMINI_API_KEY:
    raise ValueError("Missing environment variable: GEMINI_API_KEY")
if not MODEL:
    raise ValueError("Missing environment variable: MODEL")

llm = LLM(model=MODEL, api_key=GEMINI_API_KEY)


# Data Fetching Functions

def fetch_market_data() -> dict:
    """
    Fetch real-time cryptocurrency market data from the CoinMarketCap API.
    
    Returns:
        dict: JSON response containing market data (name, symbol, price, market cap, volume, etc.).
    """
    url = "https://pro-api.coinmarketcap.com/v1/cryptocurrency/listings/latest"
    params = {
        'start': '1',
        'limit': '20',
        'convert': 'USD',
        'sort': 'market_cap',
        'sort_dir': 'desc'
    }
    headers = {'X-CMC_PRO_API_KEY': COINMARKETCAP_API}
    try:
        response = requests.get(url, headers=headers, params=params)
        response.raise_for_status()
        return response.json()
    except Exception as e:
        st.error(f"Error fetching market data: {e}")
        return {}

def fetch_live_sentiment_data() -> dict:
    """
    Fetch live sentiment data from the Crypto Fear & Greed Index API.
    
    Returns:
        dict: Contains sentiment value, classification, timestamp, etc.
    """
    url = "https://api.alternative.me/fng/?limit=1"
    try:
        response = requests.get(url)
        response.raise_for_status()
        data = response.json()
        if "data" in data and data["data"]:
            return data["data"][0]
        else:
            return {}
    except Exception as e:
        st.error(f"Error fetching live sentiment data: {e}")
        return {}

def generate_sentiment_analysis() -> (dict, str):
    """
    Generate live sentiment analysis data and a textual report.
    
    Returns:
        tuple: A tuple containing the sentiment_data dictionary and a brief sentiment report.
    """
    sentiment_data = fetch_live_sentiment_data()
    if sentiment_data:
        sentiment_report = (
            f"The current Crypto Fear & Greed Index is **{sentiment_data.get('value', 'N/A')}** "
            f"({sentiment_data.get('value_classification', 'Unknown')}). "
            "This reflects the overall market sentiment as of now."
        )
    else:
        sentiment_report = "Live sentiment data is currently unavailable."
    return sentiment_data, sentiment_report


# Streamlit UI Configuration & Styling

st.set_page_config(
    page_title="AI Agent for Crypto Market Analysis",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
    <style>
    body { background-color: #0e1117; color: #d1d5db; }
    .main { background-color: #0e1117; }
    .stButton>button { background-color: #00ADB5; color: white; border: none; border-radius: 5px; padding: 0.5rem 1rem; font-size: 1rem; transition: background-color 0.3s ease, transform 0.2s ease; }
    .stButton>button:hover { background-color: #007A7F; transform: translateY(-2px); }
    .stMetric { background-color: #1a1a1a; border-radius: 10px; padding: 1rem; }
    .chart-container { background-color: #1a1a1a; border-radius: 10px; padding: 1rem; margin-top: 1rem; box-shadow: 0 4px 6px rgba(0,0,0,0.3); }
    h1, h2, h3, h4 { color: #00ADB5; }
    .stTabs [data-baseweb="tab-list"] { gap: 0.5rem; background-color: #1a1a1a; border-radius: 5px; }
    .stTabs [data-baseweb="tab"] { background-color: #333; border-radius: 5px 5px 0 0; color: #d1d5db; padding: 0.75rem 1rem; font-size: 1rem; }
    .stTabs [aria-selected="true"] { background-color: #00ADB5; color: white; }
    .sidebar .sidebar-content { background-color: #1a1a1a; color: #d1d5db; }
    </style>
""", unsafe_allow_html=True)


# Sidebar: Dynamic User Inputs

with st.sidebar:
    st.title("💹 Analysis Controls")
    with st.expander("⚙️ Settings", expanded=True):
        timeframe = st.selectbox("Analysis Timeframe", ["24H", "7D", "30D", "90D"])
        top_n = st.slider("Number of Coins for Technical Analysis", 5, 20, 10)
        update_freq = st.selectbox("Update Frequency", ["1min", "5min", "15min"])
        use_case = st.selectbox("Select Use Case", ["General Overview", "In-depth Technical Analysis", "Sentiment Focused", "Custom"])
        additional_note = st.text_input("Additional Note (optional)", "")

# Current date for inclusion in prompts and final report
today_str = datetime.now().strftime("%B %d, %Y")


# Dynamic Task Descriptions Based on User Input

market_analysis_description = (
    f"**Report Date:** {today_str}\n"
    f"1. Analyze current market conditions and trends over the past {timeframe}.\n"
    f"2. Focus on the top {top_n} cryptocurrencies by market cap.\n"
    "3. Identify key market drivers, catalysts, and risks.\n"
    "4. Evaluate overall market sentiment and momentum.\n"
    "5. Generate price predictions and risk assessments."
)
technical_analysis_description = (
    f"**Report Date:** {today_str}\n"
    f"1. Perform technical analysis on the top {top_n} cryptocurrencies over the past {timeframe}.\n"
    "2. Generate trading signals and identify chart patterns.\n"
    "3. Calculate key technical indicators (RSI, MACD, MA).\n"
    "4. Identify support and resistance levels.\n"
    "5. Provide probability-based trade recommendations."
)
sentiment_task_description = (
    f"**Report Date:** {today_str}\n"
    "1. Analyze the latest news and social media sentiment.\n"
    f"2. Summarize market sentiment trends over the past {timeframe}.\n"
    "3. Identify the impact of recent events on market sentiment."
)
report_generation_description = (
    f"**Report Date:** {today_str}\n"
    "1. Synthesize all previous analyses into a final, concise report.\n"
    "2. Create an executive summary with key findings and actionable recommendations.\n"
    "3. Highlight the most critical market insights from the data and analysis."
)
if additional_note:
    report_generation_description += f"\nNote: {additional_note}"


# CrewAI Agents and Dynamic Tasks Definition

# Define Agents (prompts are maintained; you can further tweak backstories as needed)
market_researcher = Agent(
    role="Senior Market Research Analyst",
    goal="Analyze current market conditions, trends, and provide investment insights.",
    backstory=(
        "A seasoned analyst with deep knowledge of cryptocurrency markets. "
        "Expert in identifying trends and emerging opportunities through data-driven analysis."
    ),
    tools=[SerperDevTool(n_results=5)],
    llm=llm,
    verbose=True
)

technical_analyst = Agent(
    role="Technical Analysis Specialist",
    goal="Perform technical analysis and generate trading signals.",
    backstory=(
        "A quantitative analyst with expertise in technical indicators and chart patterns. "
        "Skilled in using RSI, MACD, and moving averages to provide actionable trade recommendations."
    ),
    tools=[SerperDevTool(n_results=5)],
    llm=llm,
    verbose=True
)

news_analyst = Agent(
    role="Crypto News & Sentiment Analyst", 
    goal="Monitor news, social media, and overall market sentiment.",
    backstory=(
        "An expert in analyzing the impact of news and social media on crypto markets. "
        "Capable of identifying key events that drive market sentiment."
    ),
    tools=[SerperDevTool(n_results=5)],
    llm=llm,
    verbose=True
)

report_writer = Agent(
    role="Financial Report Writer",
    goal="Create comprehensive investment reports with actionable insights.",
    backstory=(
        "Experienced in crafting detailed financial reports and investment recommendations. "
        "Specializes in presenting complex analysis in a clear and concise manner."
    ),
    tools=[SerperDevTool(n_results=10)],
    llm=llm,
    verbose=True
)

# Update tasks with dynamic descriptions
market_analysis = Task(
    description=market_analysis_description,
    expected_output="A concise market analysis report summarizing current conditions, trends, price predictions, and risk factors.",
    agent=market_researcher
)
technical_analysis = Task(
    description=technical_analysis_description,
    expected_output="A concise technical analysis report with key trading signals, support/resistance levels, and indicator summaries.",
    agent=technical_analyst
)
sentiment_task = Task(
    description=sentiment_task_description,
    expected_output="A brief sentiment analysis summary including overall sentiment scores and key drivers.",
    agent=news_analyst
)
report_generation = Task(
    description=report_generation_description,
    expected_output=(
        "A final executive report that includes an overview of the current market, "
        "key technical insights, sentiment analysis, and recommendations for top long and short-term investment opportunities."
    ),
    agent=report_writer
)

crypto_crew = Crew(
    agents=[market_researcher, technical_analyst, news_analyst, report_writer],
    tasks=[market_analysis, technical_analysis, sentiment_task, report_generation],
    verbose=True
)


# Main Application UI and Analysis Trigger

st.title("🚀 AI Agent for Crypto Market Analysis")

if st.button("Generate Analysis"):
    with st.spinner("Fetching current market data and generating today's report..."):
        market_data = fetch_market_data()
        if not market_data.get("data"):
            st.error("No market data returned from API.")
            st.stop()

        # Build a DataFrame from the market data
        df = pd.DataFrame([{
            'name': coin['name'],
            'symbol': coin['symbol'],
            'price': coin['quote']['USD']['price'],
            'market_cap': coin['quote']['USD']['market_cap'],
            'volume_24h': coin['quote']['USD']['volume_24h'],
            'change_24h': coin['quote']['USD']['percent_change_24h']
        } for coin in market_data['data']])

        try:
            # Run the CrewAI workflow using the dynamic tasks
            ai_report = crypto_crew.kickoff()
        except Exception as e:
            st.error(f"Error during AI analysis: {e}")
            st.stop()

        # Calculate key metrics for the summary
        total_market_cap = df['market_cap'].sum() / 1e9
        total_volume = df['volume_24h'].sum() / 1e9
        avg_change = df['change_24h'].mean()
        data_summary = (
            f"**Market Summary (as of {today_str}):**\n\n"
            f"- **Total Market Cap:** ${total_market_cap:.2f}B\n"
            f"- **24h Volume:** ${total_volume:.2f}B\n"
            f"- **Average 24h Change:** {avg_change:.2f}%\n\n"
        )

        final_report = (
            f"### Executive Summary for {today_str}\n\n"
            f"{data_summary}"
            f"**Key AI Insights:**\n"
            f"{ai_report}\n\n"
            f"*Recommendation:* Monitor market drivers and adjust positions as needed for both long and short-term investments."
        )

        
        # Display Analysis Results in Tabs
        
        tab1, tab2, tab3, tab4, tab5 = st.tabs(
            ["Market Overview", "Technical Analysis", "Sentiment", "Detailed Data", "Final Report"]
        )

        with tab1:
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Total Market Cap", f"${total_market_cap:.2f}B")
            with col2:
                st.metric("24h Volume", f"${total_volume:.2f}B")
            with col3:
                st.metric("Avg 24h Change", f"{avg_change:.2f}%")
            with st.container():
                st.markdown('<div class="chart-container">', unsafe_allow_html=True)
                treemap_fig = px.treemap(
                    df,
                    path=[px.Constant("Crypto"), 'name'],
                    values='market_cap',
                    title="Market Cap Distribution"
                )
                st.plotly_chart(treemap_fig, use_container_width=True)
                st.markdown('</div>', unsafe_allow_html=True)

        with tab2:
            with st.container():
                st.markdown('<div class="chart-container">', unsafe_allow_html=True)
                price_fig = go.Figure()
                hours = list(range(24))
                for _, coin in df.head(top_n).iterrows():
                    time_points = [datetime.now() - timedelta(hours=x) for x in hours]
                    price_series = [
                        coin['price'] * (1 + coin['change_24h'] / 100 * (hour / 24))
                        for hour in hours
                    ]
                    price_fig.add_trace(go.Scatter(
                        name=coin['symbol'],
                        x=time_points,
                        y=price_series,
                        mode="lines"
                    ))
                price_fig.update_layout(
                    title="Price Performance (24H)",
                    template="plotly_dark",
                    xaxis_title="Time",
                    yaxis_title="Price (USD)"
                )
                st.plotly_chart(price_fig, use_container_width=True)
                st.markdown('</div>', unsafe_allow_html=True)

        with tab3:
            st.subheader("Market Sentiment Analysis")
            sentiment_data, sentiment_report = generate_sentiment_analysis()
            if sentiment_data:
                try:
                    gauge_fig = go.Figure(go.Indicator(
                        mode="gauge+number+delta",
                        value=float(sentiment_data.get("value", 0)),
                        delta={'reference': 50, 'increasing': {"color": "red"}, 'decreasing': {"color": "green"}},
                        title={'text': "Fear & Greed Index"},
                        gauge={
                            'axis': {'range': [0, 100]},
                            'bar': {'color': "orange"},
                            'steps': [
                                {'range': [0, 20], 'color': "red"},
                                {'range': [20, 40], 'color': "orange"},
                                {'range': [40, 60], 'color': "yellow"},
                                {'range': [60, 80], 'color': "lightgreen"},
                                {'range': [80, 100], 'color': "green"}
                            ]
                        }
                    ))
                    st.plotly_chart(gauge_fig, use_container_width=True)
                except Exception as e:
                    st.error(f"Error displaying gauge chart: {e}")
            st.markdown("### Sentiment Analysis Report")
            st.markdown(sentiment_report)

        with tab4:
            st.subheader("Detailed Cryptocurrency Data")
            st.dataframe(df.sort_values(by="market_cap", ascending=False))
            st.markdown("Use the table above to sort and search for specific coins.")

        with tab5:
            st.subheader("Final Executive Report")
            st.markdown(final_report)
else:
    st.info("Click 'Generate Analysis' to generate today's AI-powered market analysis.")

st.markdown("""
    <div style="text-align: center; margin-top: 2rem; padding: 1rem; background-color: #1a1a1a; border-radius: 10px;">
        <p>Powered by AI • Real-time Market Data • <a href="https://linkedin.com/in/heyibad" target="_blank">Made by Ibad</a> ❤️</p>
    </div>
""", unsafe_allow_html=True)
