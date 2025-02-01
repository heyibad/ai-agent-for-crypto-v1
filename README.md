# AI Agent for Crypto Analysis

An interactive web-based dashboard & AI Agent to analyze and visualize real-time cryptocurrency market data. This application integrates multiple analysis components including sentiment analysis, market performance visualization, detailed coin data, and an executive report – all built on top of Streamlit and Plotly.

---

## Features

- **Real-time Data Visualization:** View live market metrics including market cap, volume, and price changes.
- **Interactive Charts:** Experience dynamic charts such as gauge charts for sentiment analysis and line charts for price performance.
- **Detailed Cryptocurrency Data:** Sort and search for specific coins with an interactive data table.
- **Comprehensive Reports:** Get an in-depth sentiment analysis report and a final executive report with investment insights.
- **User Friendly UI:** A sleek, professional dark-themed interface with interactive tabs and controls.
- **AI Powered Analysis:** Leverages AI agents to process and report market trends and sentiment.

---

## Installation

1. Clone the Repository

    ```bash
    git clone https://github.com/heyibad/ai-agent-for-crypto-v1.git
    cd ai-agent-for-crypto-v1
    ```

2. Install Dependencies & Create a virtual environment (optional)

    ```bash
    python -m venv .venv
    source .venv/bin/activate  # On Windows use `.venv\Scripts\activate`
    pip install -r requirements.txt
    ```

3. Run the Application

    ```bash
    streamlit run app.py
    ```