#!/usr/bin/env python3
"""
Integrated End-to-End RL Pipeline for Day-Trading Stocks

This program:
  1. Loads and preprocesses historical stock data from Yahoo Finance.
     - It computes technical indicators (SMA, RSI, etc.).
     - It fetches current news sentiment from NewsAPI using VADER.
     - It appends the sentiment as an extra feature to all rows.
  2. Creates a custom Gym environment using this processed data.
  3. Implements an Actor–Critic RL agent (using PyTorch).
  4. Trains the agent over multiple episodes.
  5. Evaluates and logs performance (using TensorBoard and matplotlib).

Run this file to execute the full training and evaluation pipeline.
"""

import os
import time
import random
import datetime
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from collections import deque

import yfinance as yf
import ta  # Technical Analysis library
from newsapi import NewsApiClient
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

import gym
from gym import spaces

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter

# =============================================================================
# Data Loading & Preprocessing Functions
# =============================================================================

def fetch_stock_data(ticker: str, start_date: datetime.date, end_date: datetime.date) -> pd.DataFrame:
    """
    Fetches historical stock data from Yahoo Finance and flattens columns if necessary.
    
    Args:
        ticker (str): Stock ticker symbol (e.g., "AAPL").
        start_date (datetime.date): Start date for historical data.
        end_date (datetime.date): End date for historical data.
    
    Returns:
        pd.DataFrame: DataFrame with OHLCV data and flattened columns.
    """
    data = yf.download(ticker, start=start_date, end=end_date)
    if data.empty:
        raise ValueError("No data fetched. Check the ticker or the date range.")
    
    # Flatten MultiIndex columns if they exist
    if isinstance(data.columns, pd.MultiIndex):
        data.columns = data.columns.droplevel(1)
    
    return data

def add_technical_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """
    Adds several technical indicators to the DataFrame.
    
    Args:
        df (pd.DataFrame): DataFrame with OHLCV stock data.
    
    Returns:
        pd.DataFrame: Updated DataFrame including technical indicators.
    """
    # Ensure that the 'Close' column is a 1-D Series
    close_series = df['Close'].squeeze()
    
    # Simple Moving Average over 10 days
    df['SMA_10'] = ta.trend.sma_indicator(close=close_series, window=10)
    
    # Relative Strength Index (RSI) over 14 days
    df['RSI'] = ta.momentum.rsi(close=close_series, window=14)
    
    # Additional indicators (e.g., MACD, Bollinger Bands) can be added here.
    return df

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
                                        page_size=100)
    
    analyzer = SentimentIntensityAnalyzer()
    sentiments = []
    
    for article in response.get('articles', []):
        # Use description if available, otherwise fall back on title
        text = article.get('description') or article.get('title', '')
        if text:
            sentiment = analyzer.polarity_scores(text)
            sentiments.append(sentiment['compound'])
    
    avg_sentiment = sum(sentiments) / len(sentiments) if sentiments else 0.0
    return avg_sentiment

# =============================================================================
# Custom Gym Environment for Stock Trading
# =============================================================================

class StockTradingEnv(gym.Env):
    """
    Custom OpenAI Gym environment for day-trading a single stock.
    
    The environment uses preprocessed historical data.
    When the agent chooses to trade (action==1), it simulates buying at the open and selling at the close.
    """
    metadata = {'render.modes': ['human']}

    def __init__(self, data: pd.DataFrame, initial_cash: float = 100000,
                 transaction_cost_pct: float = 0.001):
        super(StockTradingEnv, self).__init__()

        self.data = data.copy()  # DataFrame indexed by Date with features as columns
        self.initial_cash = initial_cash
        self.transaction_cost_pct = transaction_cost_pct

        self.n_steps = len(self.data)
        self.current_step = 0

        # Action space: 0 = Do Nothing, 1 = Trade (buy at open, sell at close)
        self.action_space = spaces.Discrete(2)

        # Observation space: the state vector (all feature columns)
        self.state_dim = self.data.shape[1]
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf,
                                            shape=(self.state_dim,), dtype=np.float32)

        self.reset()

    def reset(self):
        self.cash = self.initial_cash
        self.shares_held = 0
        self.current_step = 0
        self.trades = []  # Log of trades
        return self._get_observation()

    def _get_observation(self):
        # Get the state vector for the current step as a numpy float32 array
        obs = self.data.iloc[self.current_step].values.astype(np.float32)
        return obs

    def step(self, action: int):
        """
        Execute one step in the environment.
        
        If action == 1, simulate a trade: buy at open and sell at close.
        Reward is the profit (or loss) from the trade after transaction costs.
        """
        current_data = self.data.iloc[self.current_step]
        open_price = current_data['Open']
        close_price = current_data['Close']

        reward = 0.0

        if action == 1:
            # Buy if not already holding shares
            if self.shares_held == 0:
                shares_to_buy = int(self.cash // open_price)
                if shares_to_buy > 0:
                    cost = shares_to_buy * open_price * (1 + self.transaction_cost_pct)
                    self.cash -= cost
                    self.shares_held += shares_to_buy
                    self.trades.append(('buy', self.current_step, shares_to_buy, open_price))
            # Sell at close if holding shares
            if self.shares_held > 0:
                proceeds = self.shares_held * close_price * (1 - self.transaction_cost_pct)
                self.cash += proceeds
                self.trades.append(('sell', self.current_step, self.shares_held, close_price))
                profit = proceeds - (self.shares_held * open_price)
                reward = profit
                self.shares_held = 0

        self.current_step += 1
        done = (self.current_step >= self.n_steps)
        obs = self._get_observation() if not done else np.zeros(self.state_dim, dtype=np.float32)
        info = {'cash': self.cash, 'trades': self.trades}
        return obs, reward, done, info

    def render(self, mode='human'):
        print(f"Step: {self.current_step}, Cash: {self.cash}, Trades: {self.trades}")

# =============================================================================
# Actor–Critic Network and RL Agent (Using PyTorch)
# =============================================================================

class ActorCritic(nn.Module):
    """
    Shared network that outputs both:
      - A probability distribution over actions (the policy).
      - A state-value estimation.
    """
    def __init__(self, input_dim: int, hidden_dim: int, n_actions: int):
        super(ActorCritic, self).__init__()
        self.shared_layers = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU()
        )
        self.actor = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, n_actions),
            nn.Softmax(dim=-1)
        )
        self.critic = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, x):
        shared = self.shared_layers(x)
        action_probs = self.actor(shared)
        state_value = self.critic(shared)
        return action_probs, state_value

class RLAgent:
    """
    RL Agent implementing an Advantage Actor–Critic (A2C) method.
    """
    def __init__(self, input_dim: int, hidden_dim: int, n_actions: int,
                 lr: float = 1e-3, gamma: float = 0.99):
        self.gamma = gamma
        self.model = ActorCritic(input_dim, hidden_dim, n_actions)
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

    def select_action(self, state: np.ndarray):
        state_tensor = torch.from_numpy(state).float().to(self.device)
        action_probs, state_value = self.model(state_tensor)
        dist = torch.distributions.Categorical(action_probs)
        action = dist.sample()
        return action.item(), dist.log_prob(action), state_value

    def update(self, rewards: list, log_probs: list, state_values: list,
               next_state_value: torch.Tensor, done: bool):
        """
        Compute discounted returns and perform an advantage actor–critic update.
        """
        returns = []
        R = 0 if done else next_state_value.item()
        for r in reversed(rewards):
            R = r + self.gamma * R
            returns.insert(0, R)
        returns = torch.tensor(returns).float().to(self.device)

        state_values = torch.cat(state_values).squeeze()
        log_probs = torch.stack(log_probs)
        advantage = returns - state_values

        actor_loss = - (log_probs * advantage.detach()).mean()
        critic_loss = advantage.pow(2).mean()
        loss = actor_loss + 0.5 * critic_loss

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return loss.item()

# =============================================================================
# Training & Evaluation Functions
# =============================================================================

def evaluate_agent(env: StockTradingEnv, agent: RLAgent, n_episodes: int = 5):
    """
    Evaluate the agent over multiple episodes.
    Returns the average reward.
    """
    total_rewards = []
    for _ in range(n_episodes):
        state = env.reset()
        done = False
        episode_reward = 0
        while not done:
            with torch.no_grad():
                action, _, _ = agent.select_action(state)
            state, reward, done, _ = env.step(action)
            episode_reward += reward
        total_rewards.append(episode_reward)
    return np.mean(total_rewards)

def train_agent(env: StockTradingEnv, agent: RLAgent,
                n_episodes: int = 1000, log_interval: int = 10,
                validation_env: StockTradingEnv = None):
    """
    Train the RL agent over multiple episodes.
    Uses TensorBoard for logging and periodically evaluates on a validation environment.
    """
    writer = SummaryWriter(log_dir="runs/stock_trading_rl")
    episode_rewards = []

    for episode in range(n_episodes):
        state = env.reset()
        done = False

        log_probs = []
        state_values = []
        rewards = []
        total_reward = 0

        while not done:
            action, log_prob, state_value = agent.select_action(state)
            next_state, reward, done, info = env.step(action)

            log_probs.append(log_prob)
            state_values.append(state_value)
            rewards.append(reward)
            total_reward += reward

            state = next_state

        # Bootstrap value for terminal state (if available)
        next_state_value = torch.tensor(0.0).to(agent.device)
        if not done:
            next_state_tensor = torch.from_numpy(state).float().to(agent.device)
            _, next_state_value = agent.model(next_state_tensor)

        loss = agent.update(rewards, log_probs, state_values, next_state_value, done)

        episode_rewards.append(total_reward)
        writer.add_scalar("Loss/train", loss, episode)
        writer.add_scalar("Reward/train", total_reward, episode)

        if episode % log_interval == 0:
            print(f"Episode {episode:4d}: Total Reward = {total_reward:8.2f}, Loss = {loss:.4f}")
            if validation_env is not None:
                val_reward = evaluate_agent(validation_env, agent, n_episodes=5)
                writer.add_scalar("Reward/validation", val_reward, episode)
                print(f"   Validation Average Reward: {val_reward:8.2f}")

    writer.close()
    return episode_rewards

# =============================================================================
# Main Function: Data Loading, Environment Creation, Training, and Evaluation
# =============================================================================

def main():
    # ------------------------------- Data Preprocessing -------------------------------
    ticker = "AAPL"            # Example: Apple Inc.
    company_name = "Apple"
    news_api_key = "8ff74aefdb9c4516be523ca21d39b5e2"  # Replace with your actual NewsAPI key

    today = datetime.date.today()
    start_date = today - datetime.timedelta(days=30)
    end_date = today

    # 1. Fetch historical stock data.
    try:
        stock_df = fetch_stock_data(ticker, start_date, end_date)
        print("Fetched stock data:")
        print(stock_df.tail(), "\n")
    except ValueError as e:
        print(f"Error fetching stock data: {e}")
        return

    # 2. Compute technical indicators.
    try:
        stock_df = add_technical_indicators(stock_df)
        print("Stock data with technical indicators:")
        print(stock_df.tail(), "\n")
    except Exception as e:
        print(f"Error computing technical indicators: {e}")
        return

    # 3. Fetch news sentiment for today.
    news_from_date = today.strftime("%Y-%m-%d")
    news_to_date = today.strftime("%Y-%m-%d")
    sentiment_score = fetch_news_sentiment(company_name, news_from_date, news_to_date, news_api_key)
    print(f"Average News Sentiment for {company_name} on {news_from_date}: {sentiment_score}\n")

    # 4. Add the news sentiment as a new feature to every row.
    stock_df["news_sentiment"] = sentiment_score

    # (Optional) Save processed data to CSV if you want persistence:
    processed_data_path = "processed_stock_data.csv"
    stock_df.to_csv(processed_data_path, index_label="Date")
    print(f"Processed data saved to {processed_data_path}\n")

    # ------------------------------- Environment & Agent Setup -------------------------------
    # For training, we use the full processed DataFrame.
    # (In practice, you might split data into training and validation sets.)
    train_data = stock_df.copy()
    val_data = stock_df.copy()

    train_env = StockTradingEnv(data=train_data, initial_cash=100000)
    val_env = StockTradingEnv(data=val_data, initial_cash=100000)

    input_dim = train_env.observation_space.shape[0]
    n_actions = train_env.action_space.n
    hidden_dim = 128  # Adjust network size as needed

    agent = RLAgent(input_dim=input_dim, hidden_dim=hidden_dim,
                    n_actions=n_actions, lr=1e-4, gamma=0.99)

    # ------------------------------- Training -------------------------------
    n_episodes = 1000  # Increase for more extensive training
    print("Starting training...")
    rewards = train_agent(train_env, agent, n_episodes=n_episodes,
                          log_interval=10, validation_env=val_env)

    # Save the trained model.
    torch.save(agent.model.state_dict(), "actor_critic_stock_trading.pth")
    print("Training completed and model saved as 'actor_critic_stock_trading.pth'.")

    # ------------------------------- Final Evaluation -------------------------------
    final_reward = evaluate_agent(val_env, agent, n_episodes=10)
    print(f"Final Evaluation Average Reward: {final_reward:.2f}")

    # Plot training rewards.
    plt.figure(figsize=(10, 5))
    plt.plot(rewards)
    plt.xlabel("Episode")
    plt.ylabel("Total Reward")
    plt.title("Training Rewards over Episodes")
    plt.show()

if __name__ == "__main__":
    main()
