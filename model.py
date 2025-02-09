import datetime
import yfinance as yf
import pandas as pd
import ta  # Technical Analysis library
from newsapi import NewsApiClient
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

# -------------------------------
# Function to fetch historical stock data
# -------------------------------
def fetch_stock_data(ticker: str, start_date: datetime.date, end_date: datetime.date) -> pd.DataFrame:
    """
    Fetches historical stock data from Yahoo Finance.
    
    Args:
        ticker (str): Stock ticker symbol (e.g., "AAPL").
        start_date (datetime.date): Start date for historical data.
        end_date (datetime.date): End date for historical data.
    
    Returns:
        pd.DataFrame: DataFrame with OHLCV data.
    """
    data = yf.download(ticker, start=start_date, end=end_date)
    if data.empty:
        raise ValueError("No data fetched. Check the ticker or the date range.")
    return data

# -------------------------------
# Function to add technical indicators to the stock data
# -------------------------------
def add_technical_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """
    Adds several technical indicators to the DataFrame.
    
    Args:
        df (pd.DataFrame): DataFrame with OHLCV stock data.
    
    Returns:
        pd.DataFrame: Updated DataFrame including technical indicators.
    """
    # Simple Moving Average over 10 days
    df['SMA_10'] = ta.trend.sma_indicator(df['Close'], window=10)
    
    # Relative Strength Index (RSI) over 14 days
    df['RSI'] = ta.momentum.rsi(df['Close'], window=14)
    
    # You can add more indicators here (MACD, Bollinger Bands, etc.)
    return df

# -------------------------------
# Function to fetch news articles and compute sentiment
# -------------------------------
def fetch_news_sentiment(query: str, from_date: str, to_date: str, api_key: str) -> float:
    """
    Fetches news articles using NewsAPI and calculates an average sentiment score using VADER.
    
    Args:
        query (str): Search query (e.g., company name).
        from_date (str): Start date in 'YYYY-MM-DD' format.
        to_date (str): End date in 'YYYY-MM-DD' format.
        api_key (str): API key for NewsAPI.
    
    Returns:
        float: Average compound sentiment score from the articles.
    """
    newsapi = NewsApiClient(api_key=api_key)
    response = newsapi.get_everything(q=query,
                                        from_param=from_date,
                                        to=to_date,
                                        language='en',
                                        sort_by='relevancy',
                                        page_size=100)  # maximum page size for broader coverage
    
    analyzer = SentimentIntensityAnalyzer()
    sentiments = []
    
    for article in response.get('articles', []):
        # Use the description if available, otherwise fall back on the title
        text = article.get('description') or article.get('title', '')
        if text:
            sentiment = analyzer.polarity_scores(text)
            sentiments.append(sentiment['compound'])
    
    # Calculate the average compound sentiment (range: -1 very negative to 1 very positive)
    avg_sentiment = sum(sentiments) / len(sentiments) if sentiments else 0.0
    return avg_sentiment

# -------------------------------
# Main function: load and preprocess data
# -------------------------------
def main():
    # --- Configuration ---
    ticker = "AAPL"  # Example: Apple Inc.
    company_name = "Apple"
    news_api_key = "pub_68604a926cb6537c76dce6d0616bc89b7275b"  # <-- Replace with your NewsAPI key
    
    # Define date range for historical stock data (for example, last 30 days)
    today = datetime.date.today()
    start_date = today - datetime.timedelta(days=30)
    end_date = today  # up to today
    
    # Define date range for news (using today's date for intraday state)
    news_from_date = today.strftime("%Y-%m-%d")
    news_to_date = today.strftime("%Y-%m-%d")
    
    # --- Step 1: Fetch stock data ---
    try:
        stock_df = fetch_stock_data(ticker, start_date, end_date)
        print("Fetched stock data:")
        print(stock_df.tail(), "\n")
    except ValueError as e:
        print(f"Error fetching stock data: {e}")
        return
    
    # --- Step 2: Compute technical indicators ---
    stock_df = add_technical_indicators(stock_df)
    print("Stock data with technical indicators:")
    print(stock_df.tail(), "\n")
    
    # --- Step 3: Fetch news sentiment ---
    sentiment_score = fetch_news_sentiment(company_name, news_from_date, news_to_date, news_api_key)
    print(f"Average News Sentiment for {company_name} on {news_from_date}: {sentiment_score}\n")
    
    # --- Step 4: Combine data into a state vector ---
    # Here you would typically merge technical indicators, sentiment, and other data
    # into one structured state for your RL algorithm.
    #
    # For demonstration, we will create a simple state dictionary:
    latest_data = stock_df.iloc[-1]  # use the most recent data row
    state = {
        "ticker": ticker,
        "date": str(latest_data.name.date()),
        "open": latest_data["Open"],
        "high": latest_data["High"],
        "low": latest_data["Low"],
        "close": latest_data["Close"],
        "volume": latest_data["Volume"],
        "SMA_10": latest_data["SMA_10"],
        "RSI": latest_data["RSI"],
        "news_sentiment": sentiment_score
        # Add more features as needed
    }
    
    print("Final State for the RL Agent:")
    for key, value in state.items():
        print(f"{key}: {value}")

if __name__ == "__main__":
    main()
