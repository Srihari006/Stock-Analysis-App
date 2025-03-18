import streamlit as st
from datetime import date
import yfinance as yf
from prophet import Prophet
from prophet.plot import plot_plotly
from plotly import graph_objs as go
import pandas as pd
import numpy as np
import io
import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from prophet import Prophet
import seaborn as sns

# Constants
START = "2015-01-01"
TODAY = date.today().strftime("%Y-%m-%d")
stocks = ('TSLA', 'AAPL', 'GOOGL', 'MSFT', 'AMZN', 'META', 'NVDA', 'SPY', 'NFLX', 'BABA', 'BAJAJFINSV.NS', 'ONGC.NS', 'CIPLA.NS', 'TRENT.NS', 'WIPRO.NS', 'COALINDIA.NS', 'HEROMOTOCO.NS', 'TCS.NS', 'NTPC.NS', 'APOLLOHOSP.NS', 'BRITANNIA.NS', 'BAJAJ-AUTO.NS', 'LT.NS', 'RELIANCE.NS', 'BAJFINANCE.NS', 'TATACONSUM.NS', 'ADANIENT.NS', 'MARUTI.NS', 'HDFCLIFE.NS', 'HINDALCO.NS', 'INDUSINDBK.NS', 'TATASTEEL.NS', 'NESTLEIND.NS', 'TITAN.NS', 'ULTRACEMCO.NS', 'ITC.NS', 'KOTAKBANK.NS', 'SHRIRAMFIN.NS', 'BHARTIARTL.NS')

# Streamlit App Title and Configuration
st.set_page_config(page_title="Stock Analysis App", layout="wide", initial_sidebar_state="expanded")

# Custom CSS
st.markdown("""
    <style>
        /* Dark theme for main content */
        body, .block-container {
            background-color: lightblack !important;
            color: white !important;
        }

        /* Styling for text elements */
        .stMarkdown, .stHeader, .stSubheader, .stText, .stFileUploader {
            color: white !important;
        }

        /* Sidebar theme */
        [data-testid="stSidebar"] {
            background-color: white !important;
        }
        [data-testid="stSidebar"] * {
            color: black !important;
        }

        /* Select box styles */
        .stSelectbox div[data-baseweb="select"] {
            background-color: white !important;
            color: black !important;
        }
        .stSelectbox div[data-baseweb="select"] > div {
            color: black !important;
        }
        .stSelectbox div[data-baseweb="select"] > div > div {
            background-color: white !important;
            color: black !important;
        }

        /* Advanced Prediction section */
        .advanced-prediction {
            background-color: white !important;
            color: black !important;
            margin-bottom: 20px;
        }
        .advanced-prediction * {
            color: white !important;
        }
         input[type="number"], .stNumberInput input {
            background-color: white !important;
            color: black !important;
            border: 0.5px solid black !important;
            border-radius: 5px;
            padding: 5px;
            }
            
        /* Fix the + and - buttons */
        button[data-testid="stNumberInputStepUp"], 
        button[data-testid="stNumberInputStepDown"] {
            background-color: white !important;
            color: black !important;
            border: 0.5px solid black !important;
            border-radius: 5px;
            padding: 5px;
        }

        /* Change hover effect for buttons */
        button[data-testid="stNumberInputStepUp"]:hover, 
        button[data-testid="stNumberInputStepDown"]:hover {
            background-color: lightgray !important;
        }
        
        /* Ensure the file uploader background is white */
        section[data-testid="stFileUploaderDropzone"] {
            background-color: white !important;
            color: black !important;
            border: 0.5px solid black !important;
            border-radius: 5px;
            padding: 10px;
        }

        /* Ensure text inside the dropzone is black */
        section[data-testid="stFileUploaderDropzone"] * {
            color: black !important;
        }

        /* Change hover effect for file drop zone */
        section[data-testid="stFileUploaderDropzone"]:hover {
            background-color: white !important;
        }
            
        /* File uploader styling */
        .stFileUploader {
            background-color: white !important;
            color: black !important;
            border: 0.5px solid black !important;
        }
        .stFileUploader div, .stFileUploader button {
            color: black !important;
            background-color: white !important;
        }
    </style>
    """,
    unsafe_allow_html=True,
)

st.title(":rainbow[Stock :green[Technical] :red[Analysis] & :blue[Forecast]]")

# Sidebar
with st.sidebar:
    st.image("logo.jpg", width=200)
    st.header("Stock Analysis")
    selected_stock = st.selectbox('Select Stock Symbol', stocks)
    timeframe_option = st.selectbox("Select Timeframe", ('1d', '5d', '1mo', '3mo', '6mo', '1y', '2y', '5y', '10y'))
    show_data = st.checkbox("Show Data", value=True)
    show_chart = st.checkbox("Show Chart", value=True)
    n_years = st.slider('Years of prediction (Prophet)', 1, 4)
    period = n_years * 365

    # Advanced Prediction Section with Custom Styling
    st.markdown('<div class="advanced-prediction">', unsafe_allow_html=True)
    st.header("Advanced Prediction")
    model_option = st.selectbox("Select Model", ["None", "Hybrid Model (Prophet + LSTM)"])
    num_days = st.number_input("Enter number of days for Hybrid Prediction", min_value=1, max_value=100, value=10)
    uploaded_file = st.file_uploader("Upload Stock Data (CSV) for Hybrid Model", type=['csv'])
    st.markdown('</div>', unsafe_allow_html=True)
    
# Load Data Functions
@st.cache_data
def get_stock_info(stock):
    """Fetch stock information from Yahoo Finance."""
    try:
        stock_info = yf.Ticker(stock).info
        return {
            "Company Name": stock_info.get("longName", "N/A"),
            "Sector": stock_info.get("sector", "N/A"),
            "Industry": stock_info.get("industry", "N/A"),
            "Market Cap": stock_info.get("marketCap", "N/A"),
        }
    except Exception as e:
        st.error(f"Error fetching stock details: {e}")
        return {}
    
def load_data_yfinance(stock, period):
    df = yf.Ticker(stock).history(period=period)[['Open', 'High', 'Low', 'Close', 'Volume']]
    df = df.reset_index()
    df.columns = ['time', 'open', 'high', 'low', 'close', 'volume']
    df['time'] = df['time'].dt.strftime('%Y-%m-%d')
    df['ADX_14'], df['DMP_14'], df['DMN_14'] = compute_adx(df['high'], df['low'], df['close'], 14)
    df['EMA_20'] = df['close'].ewm(span=20).mean()
    df['EMA_200'] = df['close'].ewm(span=200).mean()
    df['RSI_14'] = compute_rsi(df['close'], 14)
    return df

@st.cache_data
def load_data_prophet(stock):
    df = yf.download(stock, start=START, end=TODAY)[['Open', 'High', 'Low', 'Close', 'Volume']]
    df = df.reset_index()
    df.columns = ['Date', 'Open', 'High', 'Low', 'Close', 'Volume']
    return df

# Technical Indicators Functions
def compute_adx(high, low, close, period=14):
    tr = pd.Series(np.maximum(high - low, np.abs(high - close.shift(1)), np.abs(low - close.shift(1))))
    atr = tr.rolling(window=period).mean()
    up_move = high.diff()
    down_move = -low.diff()
    dmp = pd.Series(np.where(up_move > down_move, up_move, 0))
    dmn = pd.Series(np.where(down_move > up_move, down_move, 0))
    smoothed_dmp = dmp.rolling(window=period).sum()
    smoothed_dmn = dmn.rolling(window=period).sum()
    dx = (np.abs(smoothed_dmp - smoothed_dmn) / (smoothed_dmp + smoothed_dmn)) * 100
    adx = dx.rolling(window=period).mean()
    return adx, smoothed_dmp, smoothed_dmn

def compute_rsi(close, period=14):
    delta = close.diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    avg_gain = gain.rolling(window=period).mean()
    avg_loss = loss.rolling(window=period).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

# Emoji Functions
def get_returns_emoji(ret_val):
    return "🔴" if ret_val < 0 else "🟢"


def get_ema_emoji(ltp, ema):
    return "🔴" if ltp < ema else "🟢"

def get_rsi_emoji(rsi):
    return "🟢" if 30 < rsi < 70 else "🔴"


def get_adx_emoji(adx):
    return "🟢" if adx > 25 else "🔴"


# Chart Function
def create_chart(df, stock):
    candlestick_chart = go.Figure(data=[
        go.Candlestick(x=df['time'], open=df['open'], high=df['high'], low=df['low'], close=df['close'], name='Candlestick')
    ])
    if 'EMA_20' in df.columns and 'EMA_200' in df.columns:
        candlestick_chart.add_trace(go.Scatter(x=df['time'], y=df['EMA_20'], mode='lines', name='EMA20'))
        candlestick_chart.add_trace(go.Scatter(x=df['time'], y=df['EMA_200'], mode='lines', name='EMA200'))
    if 'close' in df.columns:
        rolling_mean = df['close'].rolling(window=20).mean()
        rolling_std = df['close'].rolling(window=20).std()
        upper_band = rolling_mean + (rolling_std * 2)
        lower_band = rolling_mean - (rolling_std * 2)
        candlestick_chart.add_trace(go.Scatter(x=df['time'], y=upper_band, mode='lines', name='Upper Band'))
        candlestick_chart.add_trace(go.Scatter(x=df['time'], y=lower_band, mode='lines', name='Lower Band'))
    candlestick_chart.update_layout(title=f'{stock} Candlestick Chart', xaxis_title='Date', yaxis_title='Price', xaxis_rangeslider_visible=False, plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)", font=dict(color="white"))
    return candlestick_chart

# Load Data
df_technical = load_data_yfinance(selected_stock, timeframe_option)
df_prophet = load_data_prophet(selected_stock)

# Display Data
if show_data:
    st.subheader(f"{selected_stock} Data")
    st.dataframe(df_technical)
    download_df = df_technical

    def convert_df(df):
        return df.to_csv().encode('utf-8')

    csv = convert_df(download_df)

    st.download_button(
        label="Download data as CSV",
        data=csv,
        file_name=f'{selected_stock}_data.csv',
        mime='text/csv',
    )

# Display Chart
if show_chart:
    st.subheader(f"{selected_stock} Chart")
    st.plotly_chart(create_chart(df_technical, selected_stock), use_container_width=True)

# Technical Analysis Metrics
st.subheader("Technical Analysis Metrics")
if not df_technical.empty:
    latest_data = df_technical.iloc[-1]
    last_close = latest_data['close']
    ema_20 = latest_data['EMA_20']
    ema_200 = latest_data['EMA_200']
    rsi = latest_data['RSI_14']
    adx = latest_data['ADX_14']
    

    returns = ((last_close - df_technical['close'].iloc[-2]) / df_technical['close'].iloc[-2]) * 100 if len(df_technical) > 1 else 0

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric(label="Last Close", value=f"{last_close:.2f}", delta=f"{returns:.2f}%")
        st.metric(label="Returns (1 Day)", value=f"{get_returns_emoji(returns)}")

    with col2:
        st.metric(label="EMA 20", value=f"{ema_20:.2f}", delta=f"{last_close - ema_20:.2f}")
        st.metric(label="EMA 20 Status", value=f"{get_ema_emoji(last_close, ema_20)}")
        st.metric(label="EMA 200", value=f"{ema_200:.2f}", delta=f"{last_close - ema_200:.2f}")
        st.metric(label="EMA 200 Status", value=f"{get_ema_emoji(last_close, ema_200)}")

    with col3:
        st.metric(label="RSI", value=f"{rsi:.2f}", delta=f"{rsi - 50:.2f}")
        st.metric(label="RSI Status", value=f"{get_rsi_emoji(rsi)}")

    with col4:
        st.metric(label="ADX", value=f"{adx:.2f}", delta=f"{adx - 25:.2f}")
        st.metric(label="ADX Status", value=f"{get_adx_emoji(adx)}")
        
    
else:
    st.error("Data not available.")

# Prophet Prediction
st.subheader("Stock Price Prediction (Prophet Model)")

df_train_prophet = df_prophet[['Date', 'Close']]
df_train_prophet = df_train_prophet.rename(columns={"Date": "ds", "Close": "y"})

m = Prophet()
m.fit(df_train_prophet)
future_prophet = m.make_future_dataframe(periods=period)
forecast_prophet = m.predict(future_prophet)

# Display DataFrame
st.subheader("Forecast Data")
st.write(forecast_prophet.tail())

st.write(f'Prophet forecast plot for {n_years} years')
fig_prophet = plot_plotly(m, forecast_prophet)
fig_prophet.update_layout(plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)", font=dict(color="white"))
st.plotly_chart(fig_prophet, use_container_width=True)

st.write("Prophet forecast components")
fig_components = m.plot_components(forecast_prophet)
st.pyplot(fig_components)

# Hybrid Model Prediction
if uploaded_file is not None:
    st.subheader("Stock Price Prediction (Hybrid Model: Prophet + LSTM)")
    df = pd.read_csv(uploaded_file)

    # Check if required columns exist
    if "time" not in df.columns or "close" not in df.columns:
        st.error("Error: 'time' or 'close' column not found in the uploaded CSV. Please check the file format.")
    else:
        df['time'] = pd.to_datetime(df['time'])
        df_hybrid = df[['time', 'close']].rename(columns={"time": "ds", "close": "y"})  # Prophet format

        st.write("### Uploaded Data Preview for Hybrid Model")
        st.write(df_hybrid.head())

        if model_option == "Hybrid Model (Prophet + LSTM)":
            st.write("### Running Prophet Model for Hybrid Approach...")

             # Train Prophet
            prophet = Prophet()
            prophet.fit(df_hybrid)  # Use df_hybrid, not df

            # Forecast using Prophet
            future = prophet.make_future_dataframe(periods=num_days)
            forecast = prophet.predict(future)
            

            # Extract Prophet trend and compute residuals
            df_hybrid['trend'] = forecast['trend'][:len(df_hybrid)]  # Ensure correct indexing
            df_hybrid['residual'] = df_hybrid['y'] - df_hybrid['trend']  # Use df_hybrid

            # Prepare Data for LSTM
            scaler = MinMaxScaler(feature_range=(0, 1))
            scaled_residuals = scaler.fit_transform(df_hybrid['residual'].values.reshape(-1, 1))


            # Create Sequences for LSTM
            def create_sequences(data, seq_length=10):
                X, y = [], []
                for i in range(len(data) - seq_length):
                    X.append(data[i : i + seq_length])
                    y.append(data[i + seq_length])
                return np.array(X), np.array(y)

            seq_length = 10
            X_train, y_train = create_sequences(scaled_residuals, seq_length)

            # Build LSTM Model
            model = Sequential([
                LSTM(50, return_sequences=True, input_shape=(seq_length, 1)),
                LSTM(50, return_sequences=False),
                Dense(25),
                Dense(1)
            ])
            model.compile(optimizer='adam', loss='mean_squared_error')

            # Train LSTM
            st.write("### Training LSTM Model...")
            model.fit(X_train, y_train, batch_size=16, epochs=50, verbose=0)

            # Predict Residuals with LSTM
            predicted_residuals = model.predict(X_train)
            predicted_residuals = scaler.inverse_transform(predicted_residuals)

            # Final Prediction = Prophet Trend + LSTM Predicted Residuals
            final_prediction = df_hybrid['trend'].iloc[seq_length:].values + predicted_residuals.flatten()

            # Extend Predictions for the future
            future_trend = forecast['trend'].iloc[len(df): len(df) + num_days].values
            future_residuals = model.predict(scaled_residuals[-seq_length:].reshape(1, seq_length, 1))
            future_residuals = scaler.inverse_transform(future_residuals).flatten()
            future_prediction = future_trend + future_residuals

            # Plot Results
            st.write("### Prediction Results")
            fig, ax = plt.subplots(figsize=(12, 6))
            ax.plot(df_hybrid['ds'].iloc[seq_length:], df_hybrid['y'].iloc[seq_length:], label="Actual Stock Price", color='blue')
            ax.plot(df_hybrid['ds'].iloc[seq_length:], final_prediction, label="Hybrid Prediction (Prophet + LSTM)", color='red')

            # Extend the red prediction line smoothly
            future_dates = pd.date_range(start=df_hybrid['ds'].iloc[-1], periods=num_days + 1, freq='D')[1:]
            ax.plot(future_dates, future_prediction, label="Future Prediction", color='green')

            ax.legend()
            st.pyplot(fig)
            
            st.subheader("Company Details")

            stock_details = get_stock_info(selected_stock)  # Fetch and cache stock details

            if stock_details:
                st.write(f"**Company Name**: {stock_details['Company Name']}")
                st.write(f"**Sector**: {stock_details['Sector']}")
                st.write(f"**Industry**: {stock_details['Industry']}")
                st.write(f"**Market Cap**: {stock_details['Market Cap']}")
            else:
                st.write("Stock details not available.")

            # Candlestick Patterns Learning Section with images
            st.subheader("Candlestick Pattern Learning Guide")

            # Bullish Engulfing Pattern
            st.markdown("### **Bullish Engulfing Pattern**")
            st.image("DIAGRAM/bullish.jpg", width=600)
            st.markdown("""
            - **Definition**: A large green candlestick engulfs a small red candlestick, signaling a potential reversal from a downtrend to an uptrend.
            - **Indication**: This pattern suggests a shift in market sentiment towards bullishness.
            - For more details, check out resources like [Investopedia Candlestick Patterns](https://www.investopedia.com/terms/d/doji.asp).
            """)

            # Bearish Engulfing Pattern
            st.markdown("### **Bearish Engulfing Pattern**")
            st.image("DIAGRAM/bearish.jpg", width=600)
            st.markdown("""
            - **Definition**: A large red candlestick engulfs a small green candlestick, signaling a potential reversal from an uptrend to a downtrend.
            - **Indication**: This pattern indicates a shift in market sentiment towards bearishness.
            - For more details, check out resources like [Investopedia Candlestick Patterns](https://www.investopedia.com/terms/b/bearishengulfingp.asp).
            """)

            # Doji Pattern
            st.markdown("### **Doji Pattern**")
            st.image("DIAGRAM/doji.jpg", width=600)
            st.markdown("""
            - **Definition**: A candlestick with a small body, where the opening and closing prices are nearly the same, signaling indecision in the market.
            - **Indication**: This pattern often appears at the top or bottom of trends, suggesting a possible reversal.
            - For more details, check out resources like [Investopedia Candlestick Patterns](https://www.investopedia.com/terms/d/doji.asp).
            """)

            # Ascending and descending staircase
            st.markdown("### **Ascending and descending staircase**")
            st.image("DIAGRAM/Ascending-and-descending-staircase.webp", width=600)
            st.markdown("""
            - **Definition**: Ascending and descending staircases are probably the most basic chart patterns. But they’re still important to know if you’re interested in identifying and trading trends.Take a look at any market, and you’ll notice that price action is rarely linear. Even in strong uptrends and downtrends, you’ll see some movement against the prevailing momentum.
            - **Indication**: When markets are forming lower lows and lower highs this can be considered a downtrend and forms a descending staircase.In an ascending staircase, a market is moving upwards. While it retraces occasionally, it is still hitting higher highs and the lows are getting higher too. This is what a bull market generally looks like, and traders will consider going long until the uptrend comes to an end. 
            - For more details, check out resources like [Investopedia Candlestick Patterns](https://www.investopedia.com/terms/t/triangle.asp).
            """)

            # Ascending triangle
            st.markdown("### **Ascending triangle**")
            st.image("DIAGRAM/Ascending triangle.png", width=600)
            st.markdown("""
            - **Definition**: An ascending triangle is a breakout pattern that forms when the price breaches the upper horizontal trendline with rising volume. It is a bullish formation.The upper trendline must be horizontal, indicating nearly identical highs, which form a resistance level. The lower trendline is rising diagonally, indicating higher lows as buyers patiently step up their bids.
            - **Indication**: In an ascending staircase, a market is moving upwards. While it retraces occasionally, it is still hitting higher highs and the lows are getting higher too. This is what a bull market generally looks like, and traders will consider going long until the uptrend comes to an end. 
            - For more details, check out resources like [Investopedia Candlestick Patterns](https://www.investopedia.com/terms/t/triangle.asp).
            """)

            # Descending triangle
            st.markdown("### **Descending triangle**")
            st.image("DIAGRAM/Descending triangle.png", width=600)
            st.markdown("""
            - **Definition**: A descending triangle is an inverted version of the ascending triangle and is considered a breakdown pattern. The lower trendline should be horizontal, connecting near identical lows. The upper trendline declines diagonally toward the apex.The breakdown occurs when the price collapses through the lower horizontal trendline support as a downtrend resumes. The lower trendline, which was support, now becomes resistance.
            - **Indication**: When markets are forming lower lows and lower highs this can be considered a downtrend and forms a descending staircase.
            - For more details, check out resources like [Investopedia Candlestick Patterns](https://www.investopedia.com/terms/t/triangle.asp).
            """)

            # Cup and handle
            st.markdown("### **Cup and handle**")
            st.image("DIAGRAM/Cup and handle.png", width=600)
            st.markdown("""
            - **Definition**: The cup-and-handle pattern is similar to a rounded bottom, except it has a second, smaller, dip after it. The second smaller curve can resemble a flag pattern if the trend lines are parallel to each other.
            - **Indication**: And like a double bottom, the cup-and-handle is a bullish reversal pattern.
            - For more details, check out resources like [Investopedia Candlestick Patterns](https://www.investopedia.com/terms/c/cupandhandle.asp).
            """)

            # Double bottom
            st.markdown("### **Double bottom**")
            st.image("DIAGRAM/Double bottom.png", width=600)
            st.markdown("""
            - **Definition**: A double bottom is, perhaps unsurprisingly, the opposite of a double top. It’s formed when a market’s price has made two attempts to break through a support level and failed. In between, there has been a temporary price rise to a level of resistance. It creates a W-shape.
            - **Indication**: The double bottom is a bullish reversal pattern because it typically signifies the end of selling pressure and a shift towards an uptrend
            - For more details, check out resources like [Investopedia Candlestick Patterns](https://www.investopedia.com/terms/d/doublebottom.asp).
            """)

            # Double top
            st.markdown("### **Double top**")
            st.image("DIAGRAM/Double top.png", width=600)
            st.markdown("""
            - **Definition**: A double top pattern is formed after a market’s price reaches two highs consecutively with small declines in between. It forms an M-shape on a chart.The double top is a bearish reversal pattern, so it’s thought that the asset’s price will fall below the support level that forms at the low point between the two highs. 
            - **Indication**: In a double top, an upwardly trending market twice tries to hit new highs. But both times, it retraces as sellers drive the price back down – a sign that bullish momentum may be on the wane.
            - For more details, check out resources like [Investopedia Candlestick Patterns](https://www.investopedia.com/terms/d/doubletop.asp).
            """)

            # Flag
            st.markdown("### **Flag**")
            st.image("DIAGRAM/Flag.png", width=600)
            st.markdown("""
            - **Definition**: A flag pattern is created when a market’s support and resistance lines run parallel to each other, either sloping upwards or downwards. It culminates in a breakout in the opposite direction to the trendlines.
            - **Indication**: In a bullish flag, both lines point downwards and a breakout through resistance signals a new uptrendIn a bearish flag, both lines point upwards and a breakout through support signals a new downtrend
            - For more details, check out resources like [Investopedia Candlestick Patterns](https://www.investopedia.com/terms/f/flag.asp).
            """)

            # Head and shoulders
            st.markdown("### **Head and shoulders**")
            st.image("DIAGRAM/Head and shoulders.png", width=600)
            st.markdown("""
            - **Definition**: The head-and-shoulders pattern is formed of three highs:The central high is the greatest, forming the head of the pattern. It’s flanked by two lower points, which make up the shoulders.All three highs should fall to the same support level – known as the neckline – and while the first two will rebound, the final attempt should break out into a downtrend.
            - **Indication**: The head-and-shoulders is a bearish reversal pattern.
            - For more details, check out resources like [Investopedia Candlestick Patterns](https://www.investopedia.com/terms/h/head-shoulders.asp).
            """)

            # Rounded top and bottom
            st.markdown("### **Rounded top and bottom**")
            st.image("DIAGRAM/Rounded top and bottom.png", width=600)
            st.markdown("""
            - **Definition**:A rounded top or bottom are both reversal patterns. A rounded top appears as an inverted U-shape, and indicates an imminent downtrend, while a rounded bottom appears as a U and occurs before an uptrend.
            - **Indication**: In a rounded top, the buying sentiment is still gaining ground at the beginning – as evidenced by the higher highs hit by the market. 
            - For more details, check out resources like [Investopedia Candlestick Patterns](https://www.investopedia.com/terms/r/roundingbottom.asp).
            """)

            # Symmetrical triangle
            st.markdown("### **Symmetrical triangle**")
            st.image("DIAGRAM/Symmetrical triangle.png", width=600)
            st.markdown("""
            - **Definition**: A symmetrical triangle is composed of a diagonal falling upper trendline and a diagonally rising lower trendline. As the price moves toward the apex, it will inevitably breach the upper trendline for a breakout and uptrend on rising prices or breach the lower trendline forming a breakdown and downtrend with falling prices.
            - **Indication**: Symmetrical triangles tend to be continuation break patterns, which means they tend to break in the direction of the initial move before the triangle forms.
            - For more details, check out resources like [Investopedia Candlestick Patterns](https://www.investopedia.com/terms/t/triangle.asp).
            """)

            # Wedge
            st.markdown("### **Wedge**")
            st.image("DIAGRAM/Wedge.png", width=600)
            st.markdown("""
            - **Definition**: A wedge pattern is similar to a flag, except that the lines tighten toward each other instead of running parallel. As the pattern progresses, it often coincides with a decline in volume.A wedge pattern can either be rising or falling. After a rising wedge pattern, the market should break out downward, passing the support level. 
            - **Indication**: For a falling wedge, the price should break through a resistance level to start an uptrend. You can open a long position at this point, or close a short one.
            - For more details, check out resources like [Investopedia Candlestick Patterns](https://www.investopedia.com/terms/w/wedge.asp).
            """)