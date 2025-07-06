# -*- coding: utf-8 -*-
#!/usr/bin/env python3
"""
State-of-the-Art RL Framework for Day-Trading Stocks

This monolithic script implements a comprehensive RL system for day trading.
It includes modules for:
  - Data ingestion and feature engineering
  - Risk management and trade simulation
  - A custom Gym environment with normalization and advanced risk controls
  - A state-of-the-art PPO agent with entropy regularization and dynamic LR scheduling
  - Distributed training using multiprocessing
  - Hyperparameter tuning via grid search
  - Extensive logging with TensorBoard and evaluation metrics

Note: This code is experimental and for research purposes only. Extensive testing,
tuning, and risk management are required before any real-money usage.
"""

#############################
# 1. Global Imports & Config
#############################

import os
import sys
import time
import math
import random
import datetime
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import logging
from collections import deque
from multiprocessing import Pool, cpu_count
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.preprocessing import StandardScaler

# Data sources and sentiment
import yfinance as yf
import ta  # Technical Analysis indicators
from newsapi import NewsApiClient
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

# Gym & RL frameworks
import gym
from gym import spaces

# PyTorch and RL agent
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter

# Set up logging for debugging and info
logging.basicConfig(level=logging.INFO, format='[%(asctime)s] %(levelname)s - %(message)s')

#############################
# 2. Data Ingestion Module
#############################

class DataLoader:
    """Handles data ingestion from multiple sources."""
    def __init__(self, ticker: str, start_date: datetime.date, end_date: datetime.date):
        self.ticker = ticker
        self.start_date = start_date
        self.end_date = end_date

    def fetch_yahoo_data(self) -> pd.DataFrame:
        logging.info(f"Fetching historical data for {self.ticker} from Yahoo Finance.")
        data = yf.download(self.ticker, start=self.start_date, end=self.end_date, interval='1d')
        if data.empty:
            raise ValueError("No data fetched from Yahoo Finance.")
        if isinstance(data.columns, pd.MultiIndex):
            data.columns = data.columns.droplevel(1)
        logging.info(f"Fetched {len(data)} days of data from {self.start_date} to {self.end_date}")
        return data

    def fetch_news_sentiment(self, query: str, news_api_key: str) -> float:
        logging.info("Fetching news sentiment from NewsAPI.")
        newsapi = NewsApiClient(api_key=news_api_key)
        today_str = datetime.date.today().strftime("%Y-%m-%d")
        response = newsapi.get_everything(q=query,
                                          from_param=today_str,
                                          to=today_str,
                                          language='en',
                                          sort_by='relevancy',
                                          page_size=100)
        analyzer = SentimentIntensityAnalyzer()
        sentiments = []
        for article in response.get('articles', []):
            text = article.get('description') or article.get('title', '')
            if text:
                sentiments.append(analyzer.polarity_scores(text)['compound'])
        avg_sentiment = sum(sentiments) / len(sentiments) if sentiments else 0.0
        logging.info(f"Average news sentiment for {query}: {avg_sentiment:.4f}")
        return avg_sentiment

#############################
# 3. Feature Engineering Module
#############################

class FeatureEngineer:
    """Generates features from raw stock data."""
    def __init__(self, df: pd.DataFrame):
        self.df = df.copy()

    def add_technical_indicators(self) -> pd.DataFrame:
        logging.info("Adding technical indicators.")
        close_series = self.df['Close'].squeeze()
        high_series = self.df['High'].squeeze()
        low_series = self.df['Low'].squeeze()
        volume_series = self.df['Volume'].squeeze()
        
        # Simple Moving Averages
        self.df['SMA_10'] = ta.trend.sma_indicator(close=close_series, window=10)
        self.df['SMA_20'] = ta.trend.sma_indicator(close=close_series, window=20)
        self.df['SMA_50'] = ta.trend.sma_indicator(close=close_series, window=50)
        
        # Exponential Moving Average
        self.df['EMA_10'] = ta.trend.ema_indicator(close=close_series, window=10)
        self.df['EMA_20'] = ta.trend.ema_indicator(close=close_series, window=20)
        
        # Relative Strength Index
        self.df['RSI'] = ta.momentum.rsi(close=close_series, window=14)
        
        # MACD
        self.df['MACD'] = ta.trend.macd(close_series)
        self.df['MACD_signal'] = ta.trend.macd_signal(close_series)
        self.df['MACD_histogram'] = ta.trend.macd_diff(close_series)
        
        # Bollinger Bands
        self.df['BB_upper'] = ta.volatility.bollinger_hband(close_series)
        self.df['BB_lower'] = ta.volatility.bollinger_lband(close_series)
        self.df['BB_width'] = ta.volatility.bollinger_wband(close_series)
        
        # Average True Range (ATR)
        self.df['ATR'] = ta.volatility.average_true_range(high=high_series,
                                                          low=low_series,
                                                          close=close_series,
                                                          window=14)
        
        # Stochastic Oscillator
        self.df['Stoch'] = ta.momentum.stoch(high=high_series,
                                             low=low_series,
                                             close=close_series,
                                             window=14, smooth_window=3)
        
        # Williams %R
        self.df['Williams_R'] = ta.momentum.williams_r(high=high_series,
                                                       low=low_series,
                                                       close=close_series,
                                                       lbp=14)
        
        # Commodity Channel Index
        self.df['CCI'] = ta.trend.cci(high=high_series,
                                      low=low_series,
                                      close=close_series,
                                      window=20)
        
        # Money Flow Index
        self.df['MFI'] = ta.volume.money_flow_index(high=high_series,
                                                     low=low_series,
                                                     close=close_series,
                                                     volume=volume_series,
                                                     window=14)
        
        # On Balance Volume
        self.df['OBV'] = ta.volume.on_balance_volume(close=close_series, volume=volume_series)
        
        # Price changes
        self.df['Price_change'] = close_series.pct_change()
        self.df['Price_change_5d'] = close_series.pct_change(periods=5)
        self.df['Price_change_10d'] = close_series.pct_change(periods=10)
        
        # Volatility
        self.df['Volatility'] = close_series.rolling(window=20).std()

        # --- New Technical Indicators ---
        # ADX (Average Directional Index)
        self.df['ADX'] = ta.trend.adx(high=high_series, low=low_series, close=close_series, window=14)
        # Parabolic SAR
        self.df['Parabolic_SAR'] = ta.trend.psar_up(high=high_series, low=low_series, close=close_series)
        # Rate of Change (ROC)
        self.df['ROC'] = ta.momentum.roc(close=close_series, window=12)
        # TRIX
        self.df['TRIX'] = ta.trend.trix(close=close_series, window=15)
        # Accumulation/Distribution Index
        self.df['ADI'] = ta.volume.acc_dist_index(high=high_series, low=low_series, close=close_series, volume=volume_series)
        return self.df

    def add_sentiment_feature(self, sentiment: float) -> pd.DataFrame:
        logging.info("Adding sentiment feature.")
        self.df['news_sentiment'] = sentiment
        return self.df

    def add_fundamental_features(self, ticker: str) -> pd.DataFrame:
        """Add fundamental features from yfinance Ticker.info as constant columns."""
        logging.info(f"Adding fundamental features for {ticker}.")
        info = yf.Ticker(ticker).info
        # Defensive: fallback to 0 if not available
        def safe_get(key):
            return info.get(key, 0)
        self.df['pe_ratio'] = safe_get('trailingPE')
        self.df['peg_ratio'] = safe_get('pegRatio')
        self.df['price_to_book'] = safe_get('priceToBook')
        self.df['eps_ttm'] = safe_get('trailingEps')
        self.df['dividend_yield'] = safe_get('dividendYield')
        self.df['market_cap'] = safe_get('marketCap')
        self.df['shares_outstanding'] = safe_get('sharesOutstanding')
        self.df['revenue_ttm'] = safe_get('totalRevenue')
        self.df['gross_margins'] = safe_get('grossMargins')
        self.df['operating_margins'] = safe_get('operatingMargins')
        self.df['return_on_equity'] = safe_get('returnOnEquity')
        self.df['debt_to_equity'] = safe_get('debtToEquity')
        self.df['current_ratio'] = safe_get('currentRatio')
        return self.df

    def add_macro_indicators(self, start_date, end_date) -> pd.DataFrame:
        """Add macro indicators (VIX, US10Y) as columns, aligned by date index."""
        logging.info("Adding macro indicators (VIX, US10Y).")
        # Fetch VIX
        vix = yf.download('^VIX', start=start_date, end=end_date, interval='1d')[['Close']]
        vix = vix.reset_index()
        vix['Date'] = pd.to_datetime(vix['Date'])
        vix = vix.set_index('Date')
        vix = vix[~vix.index.duplicated(keep='first')]
        vix.columns = ['VIX']
        vix.index.name = self.df.index.name
        # Fetch US 10Y Treasury Yield
        us10y = yf.download('^TNX', start=start_date, end=end_date, interval='1d')[['Close']]
        us10y = us10y.reset_index()
        us10y['Date'] = pd.to_datetime(us10y['Date'])
        us10y = us10y.set_index('Date')
        us10y = us10y[~us10y.index.duplicated(keep='first')]
        us10y.columns = ['US10Y']
        us10y.index.name = self.df.index.name
        # Align by date
        self.df.index = pd.to_datetime(self.df.index)
        self.df = self.df.join(vix, how='left')
        self.df = self.df.join(us10y, how='left')
        # Forward fill for any missing macro data
        self.df['VIX'] = self.df['VIX'].fillna(method='ffill')
        self.df['US10Y'] = self.df['US10Y'].fillna(method='ffill')
        return self.df

    def add_daily_news_sentiment(self, query: str, news_api_key: str) -> pd.DataFrame:
        """Add daily news sentiment as a column using NewsAPI and VADER for each date in the DataFrame."""
        logging.info(f"Adding daily news sentiment for {query}.")
        analyzer = SentimentIntensityAnalyzer()
        newsapi = NewsApiClient(api_key=news_api_key)
        sentiments = []
        for date in self.df.index:
            date_str = pd.to_datetime(date).strftime('%Y-%m-%d')
            try:
                response = newsapi.get_everything(q=query,
                                                  from_param=date_str,
                                                  to=date_str,
                                                  language='en',
                                                  sort_by='relevancy',
                                                  page_size=20)
                daily_sentiments = []
                for article in response.get('articles', []):
                    text = article.get('description') or article.get('title', '')
                    if text:
                        daily_sentiments.append(analyzer.polarity_scores(text)['compound'])
                avg_sentiment = sum(daily_sentiments) / len(daily_sentiments) if daily_sentiments else 0.0
            except Exception as e:
                logging.warning(f"Sentiment fetch failed for {date_str}: {e}")
                avg_sentiment = 0.0
            sentiments.append(avg_sentiment)
        self.df['news_sentiment'] = sentiments
        return self.df

    def clean_data(self) -> pd.DataFrame:
        logging.info("Cleaning data: dropping rows with NaNs.")
        return self.df.dropna()

#############################
# 4. Risk Management Module
#############################

class RiskManager:
    """Contains risk management functions."""
    def __init__(self, max_position: float = 0.1, stop_loss: float = 0.05, take_profit: float = 0.10):
        self.max_position = max_position   # Fraction of cash to deploy per trade
        self.stop_loss = stop_loss         # Stop-loss threshold
        self.take_profit = take_profit     # Take-profit threshold

    def compute_position_size(self, cash: float, current_price: float) -> int:
        """Calculate number of shares to buy given risk limits."""
        investable_cash = cash * self.max_position
        shares = int(investable_cash // current_price)
        return shares

    # (Placeholders for more sophisticated risk measures, e.g. volatility adjustment)

#############################
# 5. Extended Gym Environment
#############################

class ExtendedStockTradingEnv(gym.Env):
    """
    Extended Gym environment for stock trading with risk management.
    Supports normalization, advanced risk controls, and simulates slippage.
    """
    metadata = {'render.modes': ['human']}

    def __init__(self, data: pd.DataFrame, initial_cash: float = 100000,
                 transaction_cost_pct: float = 0.001, risk_manager: RiskManager = None,
                 slippage_pct: float = 0.001, normalize: bool = True, main_ticker: str = None):
        super(ExtendedStockTradingEnv, self).__init__()
        data = data.dropna()  # Ensure no missing values
        self.raw_data = data.copy()  # For actual price computations
        self.initial_cash = initial_cash
        self.transaction_cost_pct = transaction_cost_pct
        self.slippage_pct = slippage_pct
        self.risk_manager = risk_manager if risk_manager is not None else RiskManager()
        self.normalize = normalize
        self.main_ticker = main_ticker
        if self.normalize:
            self.means = data.mean()
            self.stds = data.std() + 1e-8
            self.data = (data - self.means) / self.stds
        else:
            self.data = data
        self.n_steps = len(self.data)
        self.current_step = 0
        self.cash = self.initial_cash
        self.shares_held = 0
        self.trades = []
        self.portfolio_values = []
        self.buy_price = 0
        self.action_space = spaces.Discrete(3)  # 0: Hold, 1: Buy, 2: Sell
        self.state_dim = self.data.shape[1]
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf,
                                            shape=(self.state_dim,), dtype=np.float32)

    def reset(self):
        self.current_step = 0
        self.cash = self.initial_cash
        self.shares_held = 0
        self.trades = []
        self.portfolio_values = []
        self.buy_price = 0
        return self._get_observation()

    def _get_observation(self):
        return self.data.iloc[self.current_step].values.astype(np.float32)

    def step(self, action: int):
        # Use main ticker columns for price
        open_col = f'Open_{self.main_ticker}' if self.main_ticker else 'Open'
        close_col = f'Close_{self.main_ticker}' if self.main_ticker else 'Close'
        current_row = self.raw_data.iloc[self.current_step]
        open_price = current_row[open_col]
        close_price = current_row[close_col]
        reward = 0.0

        # Simulate slippage by perturbing price slightly
        open_price *= (1 + random.uniform(-self.slippage_pct, self.slippage_pct))
        close_price *= (1 + random.uniform(-self.slippage_pct, self.slippage_pct))

        # Calculate current portfolio value
        portfolio_value = self.cash + (self.shares_held * close_price)
        self.portfolio_values.append(portfolio_value)

        if action == 1:  # Buy
            if self.shares_held == 0:  # Only buy if not holding
                shares_to_buy = self.risk_manager.compute_position_size(self.cash, open_price)
                if shares_to_buy > 0:
                    cost = shares_to_buy * open_price * (1 + self.transaction_cost_pct)
                    self.cash -= cost
                    self.shares_held += shares_to_buy
                    self.buy_price = open_price
                    self.trades.append(('buy', self.current_step, shares_to_buy, open_price))

        elif action == 2:  # Sell
            if self.shares_held > 0:  # Only sell if holding
                proceeds = self.shares_held * close_price * (1 - self.transaction_cost_pct)
                self.cash += proceeds
                profit = proceeds - (self.shares_held * self.buy_price)
                reward = profit
                self.trades.append(('sell', self.current_step, self.shares_held, close_price))
                self.shares_held = 0
                self.buy_price = 0

        # Calculate daily return
        if len(self.portfolio_values) > 1:
            daily_return = (portfolio_value - self.portfolio_values[-2]) / self.portfolio_values[-2]
            reward += daily_return * 100  # Scale up the reward

        self.current_step += 1
        done = (self.current_step >= self.n_steps)
        obs = self._get_observation() if not done else np.zeros(self.state_dim, dtype=np.float32)
        info = {
            'cash': self.cash, 
            'trades': self.trades,
            'portfolio_value': portfolio_value,
            'shares_held': self.shares_held
        }
        return obs, reward, done, info

    def render(self, mode='human'):
        print(f"Step: {self.current_step}, Cash: {self.cash:.2f}, Shares: {self.shares_held}, Portfolio: {self.cash + (self.shares_held * self.raw_data.iloc[self.current_step][close_col]):.2f}")

#############################
# 6. PPO Agent Module
#############################

class PPOAgent(nn.Module):
    """State-of-the-art PPO network with separate actor and critic heads."""
    def __init__(self, input_dim: int, hidden_dim: int, n_actions: int):
        super(PPOAgent, self).__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.shared = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1)
        )
        self.actor = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, n_actions),
            nn.Softmax(dim=-1)
        )
        self.critic = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1)
        )

    def forward(self, x):
        shared_out = self.shared(x)
        action_probs = self.actor(shared_out)
        state_value = self.critic(shared_out)
        return action_probs, state_value

class PPOTrainer:
    """Trainer class for PPO with clipping, entropy bonus, and adaptive learning rate."""
    def __init__(self, env: gym.Env, agent: PPOAgent, lr: float = 1e-4, gamma: float = 0.99,
                 clip_epsilon: float = 0.2, entropy_coef: float = 0.01, epochs: int = 4):
        self.env = env
        self.agent = agent
        self.gamma = gamma
        self.clip_epsilon = clip_epsilon
        self.entropy_coef = entropy_coef
        self.epochs = epochs
        self.optimizer = optim.Adam(self.agent.parameters(), lr=lr)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.agent.to(self.device)
        self.lr_scheduler = optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, mode='max', patience=10, factor=0.5)

    def collect_trajectories(self, n_steps: int):
        """Collect trajectories by running the current policy."""
        states, actions, rewards, log_probs, dones = [], [], [], [], []
        state = self.env.reset()
        for _ in range(n_steps):
            state_tensor = torch.from_numpy(state).float().to(self.device)
            action_probs, _ = self.agent(state_tensor)
            dist = torch.distributions.Categorical(action_probs)
            action = dist.sample()
            log_prob = dist.log_prob(action)
            next_state, reward, done, info = self.env.step(action.item())
            states.append(state)
            actions.append(action.item())
            rewards.append(reward)
            log_probs.append(log_prob.item())
            dones.append(done)
            state = next_state
            if done:
                state = self.env.reset()
        return states, actions, rewards, log_probs, dones

    def compute_returns(self, rewards, dones):
        """Compute discounted returns."""
        returns = []
        R = 0
        for r, done in zip(reversed(rewards), reversed(dones)):
            if done:
                R = 0
            R = r + self.gamma * R
            returns.insert(0, R)
        return returns

    def ppo_update(self, trajectories, batch_size: int = 64):
        """Perform PPO update over collected trajectories."""
        states, actions, rewards, old_log_probs, dones = trajectories
        returns = self.compute_returns(rewards, dones)
        returns = torch.tensor(returns).float().to(self.device)
        states = torch.tensor(np.array(states)).float().to(self.device)
        actions = torch.tensor(actions).to(self.device)
        old_log_probs = torch.tensor(old_log_probs).float().to(self.device)

        # Normalize returns
        returns = (returns - returns.mean()) / (returns.std() + 1e-8)

        for epoch in range(self.epochs):
            for i in range(0, len(states), batch_size):
                batch_states = states[i:i+batch_size]
                batch_actions = actions[i:i+batch_size]
                batch_old_log_probs = old_log_probs[i:i+batch_size]
                batch_returns = returns[i:i+batch_size]

                action_probs, state_values = self.agent(batch_states)
                dist = torch.distributions.Categorical(action_probs)
                new_log_probs = dist.log_prob(batch_actions)
                entropy = dist.entropy().mean()

                ratio = torch.exp(new_log_probs - batch_old_log_probs)
                advantage = batch_returns - state_values.squeeze()

                surr1 = ratio * advantage
                surr2 = torch.clamp(ratio, 1 - self.clip_epsilon, 1 + self.clip_epsilon) * advantage
                actor_loss = -torch.min(surr1, surr2).mean()
                critic_loss = nn.MSELoss()(state_values.squeeze(), batch_returns)
                loss = actor_loss + 0.5 * critic_loss - self.entropy_coef * entropy

                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.agent.parameters(), 0.5)
                self.optimizer.step()

    def train(self, total_timesteps: int, log_interval: int = 100):
        writer = SummaryWriter(log_dir="runs/ppo_trading_rl")
        timestep = 0
        all_rewards = []
        while timestep < total_timesteps:
            traj = self.collect_trajectories(n_steps=256)
            states, actions, rewards, old_log_probs, dones = traj
            timestep += len(rewards)
            self.ppo_update(traj)
            episode_reward = sum(rewards)
            all_rewards.append(episode_reward)
            writer.add_scalar("Reward", episode_reward, timestep)
            print(f"Timestep: {timestep:6d} - Episode Reward: {episode_reward:.2f}")
            # Adjust learning rate if needed based on performance (dummy scheduler here)
            self.lr_scheduler.step(episode_reward)
        writer.close()
        return all_rewards

#############################
# 7. Enhanced Evaluation Module
#############################

class EnhancedEvaluator:
    """Enhanced evaluator with comprehensive metrics and visualization."""
    def __init__(self, env, agent, n_episodes: int = 10):
        self.env = env
        self.agent = agent
        self.n_episodes = n_episodes

    def evaluate(self):
        all_rewards = []
        all_actions = []
        all_portfolio_values = []
        all_trades = []
        
        for episode in range(self.n_episodes):
            state = self.env.reset()
            done = False
            ep_reward = 0
            ep_actions = []
            ep_portfolio_values = []
            
            while not done:
                state_tensor = torch.from_numpy(state).float().to(self.agent.device)
                with torch.no_grad():
                    action_probs, _ = self.agent(state_tensor)
                dist = torch.distributions.Categorical(action_probs)
                action = dist.sample().item()
                
                state, reward, done, info = self.env.step(action)
                ep_reward += reward
                ep_actions.append(action)
                ep_portfolio_values.append(info['portfolio_value'])
                
            all_rewards.append(ep_reward)
            all_actions.extend(ep_actions)
            all_portfolio_values.append(ep_portfolio_values)
            all_trades.extend(self.env.trades)
        
        # Calculate metrics
        avg_reward = np.mean(all_rewards)
        std_reward = np.std(all_rewards)
        
        # Action distribution
        action_counts = np.bincount(all_actions, minlength=3)
        action_distribution = action_counts / len(all_actions)
        
        # Trading metrics
        buy_trades = [t for t in all_trades if t[0] == 'buy']
        sell_trades = [t for t in all_trades if t[0] == 'sell']
        
        total_trades = len(buy_trades) + len(sell_trades)
        if total_trades > 0:
            trade_frequency = total_trades / self.n_episodes
        else:
            trade_frequency = 0
            
        logging.info(f"Evaluation over {self.n_episodes} episodes:")
        logging.info(f"  Average Reward: {avg_reward:.2f} ± {std_reward:.2f}")
        logging.info(f"  Action Distribution - Hold: {action_distribution[0]:.2%}, Buy: {action_distribution[1]:.2%}, Sell: {action_distribution[2]:.2%}")
        logging.info(f"  Total Trades: {total_trades}")
        logging.info(f"  Trade Frequency: {trade_frequency:.2f} trades per episode")
        
        return {
            'avg_reward': avg_reward,
            'std_reward': std_reward,
            'action_distribution': action_distribution,
            'total_trades': total_trades,
            'trade_frequency': trade_frequency,
            'portfolio_values': all_portfolio_values,
            'all_rewards': all_rewards
        }

class BacktestEvaluator:
    """Backtest the model on unseen data with comprehensive analysis."""
    def __init__(self, test_env, agent):
        self.test_env = test_env
        self.agent = agent
        
    def backtest(self):
        """Run backtest and return comprehensive results."""
        state = self.test_env.reset()
        done = False
        total_reward = 0
        portfolio_values = []
        actions_taken = []
        trades = []
        
        while not done:
            state_tensor = torch.from_numpy(state).float().to(self.agent.device)
            with torch.no_grad():
                action_probs, _ = self.agent(state_tensor)
            dist = torch.distributions.Categorical(action_probs)
            action = dist.sample().item()
            
            state, reward, done, info = self.test_env.step(action)
            total_reward += reward
            portfolio_values.append(info['portfolio_value'])
            actions_taken.append(action)
            
        trades = self.test_env.trades
        
        # Calculate backtest metrics
        initial_value = self.test_env.initial_cash
        final_value = portfolio_values[-1] if portfolio_values else initial_value
        total_return = (final_value - initial_value) / initial_value * 100
        
        # Calculate Sharpe ratio (simplified)
        if len(portfolio_values) > 1:
            returns = np.diff(portfolio_values) / portfolio_values[:-1]
            sharpe_ratio = np.mean(returns) / (np.std(returns) + 1e-8) * np.sqrt(252)
        else:
            sharpe_ratio = 0
            
        # Calculate max drawdown
        peak = initial_value
        max_drawdown = 0
        for value in portfolio_values:
            if value > peak:
                peak = value
            drawdown = (peak - value) / peak
            max_drawdown = max(max_drawdown, drawdown)
        
        return {
            'total_reward': total_reward,
            'total_return_pct': total_return,
            'final_value': final_value,
            'initial_value': initial_value,
            'portfolio_values': portfolio_values,
            'actions_taken': actions_taken,
            'trades': trades,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown,
            'trade_count': len(trades)
        }

#############################
# 8. Visualization Module
#############################

class TradingVisualizer:
    """Comprehensive visualization for trading results."""
    
    @staticmethod
    def plot_training_results(rewards, save_path="training_results.png"):
        """Plot training rewards over time."""
        plt.figure(figsize=(12, 6))
        plt.plot(rewards, alpha=0.7, linewidth=1)
        plt.xlabel("Training Episodes")
        plt.ylabel("Episode Reward")
        plt.title("PPO Training Progress")
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
        
    @staticmethod
    def plot_backtest_results(backtest_results, test_data, save_path="backtest_results.png", main_ticker=None):
        """Plot comprehensive backtest results."""
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        portfolio_values = backtest_results['portfolio_values']
        dates = test_data.index[:len(portfolio_values)]
        close_col = f'Close_{main_ticker}' if main_ticker else 'Close'
        stock_prices = test_data[close_col].values[:len(portfolio_values)]
        axes[0, 0].plot(dates, portfolio_values, linewidth=2, color='blue')
        axes[0, 0].set_title('Portfolio Value Over Time')
        axes[0, 0].set_xlabel('Date')
        axes[0, 0].set_ylabel('Portfolio Value ($)')
        axes[0, 0].grid(True, alpha=0.3)
        ax2 = axes[0, 1].twinx()
        axes[0, 1].plot(dates, portfolio_values, linewidth=2, color='blue', label='Portfolio')
        ax2.plot(dates, stock_prices, linewidth=2, color='red', alpha=0.7, label='Stock Price')
        axes[0, 1].set_title('Portfolio vs Stock Price')
        axes[0, 1].set_xlabel('Date')
        axes[0, 1].set_ylabel('Portfolio Value ($)', color='blue')
        ax2.set_ylabel('Stock Price ($)', color='red')
        axes[0, 1].grid(True, alpha=0.3)
        actions = backtest_results['actions_taken']
        action_names = ['Hold', 'Buy', 'Sell']
        action_counts = np.bincount(actions, minlength=3)
        axes[1, 0].bar(action_names, action_counts, color=['gray', 'green', 'red'])
        axes[1, 0].set_title('Action Distribution')
        axes[1, 0].set_ylabel('Count')
        trades = backtest_results['trades']
        if trades:
            trade_dates = [test_data.index[t[1]] for t in trades]
            trade_prices = [t[3] for t in trades]
            trade_types = [t[0] for t in trades]
            buy_dates = [d for d, t in zip(trade_dates, trade_types) if t == 'buy']
            buy_prices = [p for p, t in zip(trade_prices, trade_types) if t == 'buy']
            sell_dates = [d for d, t in zip(trade_dates, trade_types) if t == 'sell']
            sell_prices = [p for p, t in zip(trade_prices, trade_types) if t == 'sell']
            axes[1, 1].plot(dates, stock_prices, linewidth=1, color='black', alpha=0.7)
            axes[1, 1].scatter(buy_dates, buy_prices, color='green', s=50, marker='^', label='Buy')
            axes[1, 1].scatter(sell_dates, sell_prices, color='red', s=50, marker='v', label='Sell')
            axes[1, 1].set_title('Trading Activity')
            axes[1, 1].set_xlabel('Date')
            axes[1, 1].set_ylabel('Stock Price ($)')
            axes[1, 1].legend()
            axes[1, 1].grid(True, alpha=0.3)
        else:
            axes[1, 1].text(0.5, 0.5, 'No trades executed', ha='center', va='center', transform=axes[1, 1].transAxes)
            axes[1, 1].set_title('Trading Activity')
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
        
    @staticmethod
    def plot_evaluation_metrics(eval_results, save_path="evaluation_metrics.png"):
        """Plot evaluation metrics."""
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        # Reward distribution
        rewards = eval_results['all_rewards']
        axes[0, 0].hist(rewards, bins=20, alpha=0.7, color='blue')
        axes[0, 0].axvline(np.mean(rewards), color='red', linestyle='--', label=f'Mean: {np.mean(rewards):.2f}')
        axes[0, 0].set_title('Reward Distribution')
        axes[0, 0].set_xlabel('Reward')
        axes[0, 0].set_ylabel('Frequency')
        axes[0, 0].legend()
        
        # Action distribution
        action_dist = eval_results['action_distribution']
        action_names = ['Hold', 'Buy', 'Sell']
        colors = ['gray', 'green', 'red']
        axes[0, 1].pie(action_dist, labels=action_names, autopct='%1.1f%%', colors=colors)
        axes[0, 1].set_title('Action Distribution')
        
        # Portfolio values over episodes
        portfolio_values = eval_results['portfolio_values']
        if portfolio_values:
            final_values = [pv[-1] if pv else 0 for pv in portfolio_values]
            axes[1, 0].plot(final_values, marker='o', alpha=0.7)
            axes[1, 0].set_title('Final Portfolio Values by Episode')
            axes[1, 0].set_xlabel('Episode')
            axes[1, 0].set_ylabel('Final Portfolio Value ($)')
            axes[1, 0].grid(True, alpha=0.3)
        
        # Trading frequency
        axes[1, 1].bar(['Trades per Episode'], [eval_results['trade_frequency']], color='orange')
        axes[1, 1].set_title('Trading Frequency')
        axes[1, 1].set_ylabel('Trades per Episode')
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()

    @staticmethod
    def plot_actual_vs_predicted(backtest_results, test_data, save_path="actual_vs_predicted.png", main_ticker=None):
        """Plot actual stock price (red) vs. agent's portfolio value (green) for the test period."""
        portfolio_values = backtest_results['portfolio_values']
        dates = test_data.index[:len(portfolio_values)]
        close_col = f'Close_{main_ticker}' if main_ticker else 'Close'
        actual_prices = test_data[close_col].values[:len(portfolio_values)]
        
        plt.figure(figsize=(14, 7))
        plt.plot(dates, actual_prices, color='red', label='Actual Close Price', linewidth=2)
        plt.plot(dates, portfolio_values, color='green', label="Agent's Portfolio Value", linewidth=2)
        plt.title('Actual Stock Price vs. Agent Portfolio Value (Test Period)')
        plt.xlabel('Date')
        plt.ylabel('Value ($)')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()

    @staticmethod
    def plot_result_graph(backtest_results, test_data, save_path="result.png", main_ticker=None):
        """Plot predicted (dark green) and actual (red) for the 6-month test period."""
        portfolio_values = backtest_results['portfolio_values']
        dates = test_data.index[:len(portfolio_values)]
        close_col = f'Close_{main_ticker}' if main_ticker else 'Close'
        actual_prices = test_data[close_col].values[:len(portfolio_values)]
        
        plt.figure(figsize=(14, 7))
        plt.plot(dates, actual_prices, color='red', label='Actual Close Price', linewidth=2)
        plt.plot(dates, portfolio_values, color='#006400', label="Predicted Portfolio Value", linewidth=2)  # dark green
        plt.title('Predicted (Model) vs Actual Value (6-Month Unseen Test)')
        plt.xlabel('Date')
        plt.ylabel('Value ($)')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()

#############################
# 9. Main Function: Enhanced Implementation
#############################

def merge_ticker_features(ticker_dfs):
    """Merge a dict of {ticker: df} into a single DataFrame with suffixes."""
    merged = None
    for ticker, df in ticker_dfs.items():
        df = df.add_suffix(f'_{ticker}')
        if merged is None:
            merged = df
        else:
            merged = merged.join(df, how='outer')
    return merged

def main():
    # Configuration
    tickers = ["AAPL", "MSFT", "SPY"]  # Add more tickers as desired
    company_names = {"AAPL": "Apple", "MSFT": "Microsoft", "SPY": "S&P 500 ETF"}
    news_api_key = "8ff74aefdb9c4516be523ca21d39b5e2"  # Replace with your valid key
    
    # Date ranges: 5 years for training, 6 months for testing
    today = datetime.date.today()
    test_end_date = today
    test_start_date = today - datetime.timedelta(days=180)  # 6 months
    train_end_date = test_start_date
    train_start_date = train_end_date - datetime.timedelta(days=5*365)  # 5 years
    
    logging.info(f"Training period: {train_start_date} to {train_end_date}")
    logging.info(f"Testing period: {test_start_date} to {test_end_date}")
    
    # Fetch and process data for each ticker
    train_ticker_dfs = {}
    test_ticker_dfs = {}
    for ticker in tickers:
        logging.info(f"Fetching training data for {ticker}...")
        train_data_loader = DataLoader(ticker=ticker, start_date=train_start_date, end_date=train_end_date)
        train_raw_data = train_data_loader.fetch_yahoo_data()
        train_fe = FeatureEngineer(train_raw_data)
        train_fe.add_technical_indicators()
        train_fe.add_fundamental_features(ticker)
        train_fe.add_macro_indicators(train_start_date, train_end_date)
        # Use neutral sentiment for training
        train_fe.add_sentiment_feature(0.0)
        train_clean = train_fe.clean_data()
        train_ticker_dfs[ticker] = train_clean
        
        logging.info(f"Fetching test data for {ticker}...")
        test_data_loader = DataLoader(ticker=ticker, start_date=test_start_date, end_date=test_end_date)
        test_raw_data = test_data_loader.fetch_yahoo_data()
        test_fe = FeatureEngineer(test_raw_data)
        test_fe.add_technical_indicators()
        test_fe.add_fundamental_features(ticker)
        test_fe.add_macro_indicators(test_start_date, test_end_date)
        # Use daily news sentiment for the main ticker, else neutral
        if ticker == tickers[0]:
            test_fe.add_daily_news_sentiment(company_names[ticker], news_api_key)
        else:
            test_fe.add_sentiment_feature(0.0)
        test_clean = test_fe.clean_data()
        test_ticker_dfs[ticker] = test_clean
    
    # Merge all tickers' features
    train_merged = merge_ticker_features(train_ticker_dfs)
    test_merged = merge_ticker_features(test_ticker_dfs)
    
    # Save processed data
    train_merged.to_csv("train_stock_data.csv", index_label="Date")
    test_merged.to_csv("test_stock_data.csv", index_label="Date")
    logging.info(f"Training data: {len(train_merged)} days")
    logging.info(f"Test data: {len(test_merged)} days")
    
    # Create Risk Manager
    risk_manager = RiskManager(max_position=0.1, stop_loss=0.05, take_profit=0.10)
    
    # Create training environment
    main_ticker = tickers[0]
    train_env = ExtendedStockTradingEnv(data=train_merged, initial_cash=100000,
                                        transaction_cost_pct=0.001, risk_manager=risk_manager,
                                        slippage_pct=0.001, normalize=True, main_ticker=main_ticker)
    # Create test environment
    test_env = ExtendedStockTradingEnv(data=test_merged, initial_cash=100000,
                                       transaction_cost_pct=0.001, risk_manager=risk_manager,
                                       slippage_pct=0.001, normalize=True, main_ticker=main_ticker)
    
    # Train PPO agent
    input_dim = train_env.observation_space.shape[0]
    n_actions = train_env.action_space.n
    hidden_dim = 256
    
    ppo_agent = PPOAgent(input_dim=input_dim, hidden_dim=hidden_dim, n_actions=n_actions)
    ppo_trainer = PPOTrainer(env=train_env, agent=ppo_agent, lr=1e-4, gamma=0.99,
                             clip_epsilon=0.2, entropy_coef=0.01, epochs=4)
    
    total_timesteps = 20000  # Increased for better training
    logging.info("Starting PPO training...")
    training_rewards = ppo_trainer.train(total_timesteps=total_timesteps, log_interval=500)
    
    # Save the trained model
    model_save_path = "ppo_trading_model.pth"
    torch.save(ppo_agent.state_dict(), model_save_path)
    logging.info(f"Trained PPO model saved to {model_save_path}")
    
    # Evaluate on training data
    logging.info("Evaluating on training data...")
    train_evaluator = EnhancedEvaluator(env=train_env, agent=ppo_agent, n_episodes=5)
    train_eval_results = train_evaluator.evaluate()
    
    # Backtest on test data
    logging.info("Backtesting on test data...")
    backtest_evaluator = BacktestEvaluator(test_env=test_env, agent=ppo_agent)
    backtest_results = backtest_evaluator.backtest()
    
    # Print comprehensive results
    print("\n" + "="*60)
    print("COMPREHENSIVE TRADING RESULTS")
    print("="*60)
    
    print(f"\nTRAINING RESULTS:")
    print(f"  Average Reward: {train_eval_results['avg_reward']:.2f} ± {train_eval_results['std_reward']:.2f}")
    print(f"  Total Trades: {train_eval_results['total_trades']}")
    print(f"  Trade Frequency: {train_eval_results['trade_frequency']:.2f} trades per episode")
    
    print(f"\nBACKTEST RESULTS (6-month unseen data):")
    print(f"  Total Return: {backtest_results['total_return_pct']:.2f}%")
    print(f"  Final Portfolio Value: ${backtest_results['final_value']:,.2f}")
    print(f"  Sharpe Ratio: {backtest_results['sharpe_ratio']:.3f}")
    print(f"  Max Drawdown: {backtest_results['max_drawdown']:.2%}")
    print(f"  Total Trades: {backtest_results['trade_count']}")
    print(f"  Total Reward: {backtest_results['total_reward']:.2f}")
    
    # Calculate accuracy metrics
    if backtest_results['trade_count'] > 0:
        profitable_trades = sum(1 for i in range(0, len(backtest_results['trades']), 2) 
                              if i+1 < len(backtest_results['trades']) and 
                              backtest_results['trades'][i+1][3] > backtest_results['trades'][i][3])
        accuracy = profitable_trades / (backtest_results['trade_count'] // 2) if backtest_results['trade_count'] > 0 else 0
        print(f"  Trade Accuracy: {accuracy:.2%}")
    
    # Visualizations
    logging.info("Generating visualizations...")
    
    # Plot training results
    TradingVisualizer.plot_training_results(training_rewards, "training_results.png")
    
    # Plot backtest results
    TradingVisualizer.plot_backtest_results(backtest_results, test_merged, "backtest_results.png", main_ticker=main_ticker)
    
    # Plot evaluation metrics
    TradingVisualizer.plot_evaluation_metrics(train_eval_results, "evaluation_metrics.png")
    
    # Plot actual vs predicted (agent portfolio value)
    TradingVisualizer.plot_actual_vs_predicted(backtest_results, test_merged, "actual_vs_predicted.png", main_ticker=main_ticker)
    
    # Plot result graph (predicted vs actual, dark green)
    TradingVisualizer.plot_result_graph(backtest_results, test_merged, "result.png", main_ticker=main_ticker)
    
    print(f"\nVisualizations saved:")
    print(f"  - training_results.png")
    print(f"  - backtest_results.png") 
    print(f"  - evaluation_metrics.png")
    
    # Suggestions for improvement
    print(f"\n" + "="*60)
    print("SUGGESTIONS FOR IMPROVEMENT")
    print("="*60)
    print("1. Feature Engineering:")
    print("   - Add more technical indicators (ADX, Parabolic SAR, etc.)")
    print("   - Include market sentiment data (VIX, sector performance)")
    print("   - Add fundamental data (P/E ratios, earnings dates)")
    print("   - Include macroeconomic indicators")
    
    print("\n2. Model Architecture:")
    print("   - Try LSTM/GRU layers for temporal dependencies")
    print("   - Implement attention mechanisms")
    print("   - Use ensemble methods (multiple agents)")
    print("   - Add more sophisticated reward functions")
    
    print("\n3. Training Improvements:")
    print("   - Increase training timesteps (100k+)")
    print("   - Use curriculum learning")
    print("   - Implement experience replay")
    print("   - Add regularization techniques")
    
    print("\n4. Risk Management:")
    print("   - Implement dynamic position sizing")
    print("   - Add stop-loss and take-profit logic")
    print("   - Include portfolio diversification")
    print("   - Add volatility-adjusted position sizing")
    
    print("\n5. Data Quality:")
    print("   - Use higher frequency data (1-minute bars)")
    print("   - Include more stocks for diversification")
    print("   - Add real-time news sentiment")
    print("   - Include options data for volatility signals")

if __name__ == "__main__":
    main()
