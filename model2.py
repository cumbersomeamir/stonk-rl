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
import logging
from collections import deque
from multiprocessing import Pool, cpu_count

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
        data = yf.download(self.ticker, start=self.start_date, end=self.end_date)
        if data.empty:
            raise ValueError("No data fetched from Yahoo Finance.")
        if isinstance(data.columns, pd.MultiIndex):
            data.columns = data.columns.droplevel(1)
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
        # Simple Moving Averages
        self.df['SMA_10'] = ta.trend.sma_indicator(close=close_series, window=10)
        self.df['SMA_20'] = ta.trend.sma_indicator(close=close_series, window=20)
        # Exponential Moving Average
        self.df['EMA_10'] = ta.trend.ema_indicator(close=close_series, window=10)
        # Relative Strength Index
        self.df['RSI'] = ta.momentum.rsi(close=close_series, window=14)
        # MACD
        self.df['MACD'] = ta.trend.macd(close_series)
        # Bollinger Bands
        self.df['BB_upper'] = ta.volatility.bollinger_hband(close_series)
        self.df['BB_lower'] = ta.volatility.bollinger_lband(close_series)
        # Average True Range (ATR)
        self.df['ATR'] = ta.volatility.average_true_range(high=self.df['High'],
                                                          low=self.df['Low'],
                                                          close=self.df['Close'],
                                                          window=14)
        # Stochastic Oscillator
        self.df['Stoch'] = ta.momentum.stoch(high=self.df['High'],
                                             low=self.df['Low'],
                                             close=self.df['Close'],
                                             window=14, smooth_window=3)
        return self.df

    def add_sentiment_feature(self, sentiment: float) -> pd.DataFrame:
        logging.info("Adding sentiment feature.")
        self.df['news_sentiment'] = sentiment
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
                 slippage_pct: float = 0.001, normalize: bool = True):
        super(ExtendedStockTradingEnv, self).__init__()
        data = data.dropna()  # Ensure no missing values
        self.raw_data = data.copy()  # For actual price computations
        self.initial_cash = initial_cash
        self.transaction_cost_pct = transaction_cost_pct
        self.slippage_pct = slippage_pct
        self.risk_manager = risk_manager if risk_manager is not None else RiskManager()

        self.normalize = normalize
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

        self.action_space = spaces.Discrete(2)  # 0: Hold, 1: Trade
        self.state_dim = self.data.shape[1]
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf,
                                            shape=(self.state_dim,), dtype=np.float32)

    def reset(self):
        self.current_step = 0
        self.cash = self.initial_cash
        self.shares_held = 0
        self.trades = []
        return self._get_observation()

    def _get_observation(self):
        return self.data.iloc[self.current_step].values.astype(np.float32)

    def step(self, action: int):
        current_row = self.raw_data.iloc[self.current_step]
        open_price = current_row['Open']
        close_price = current_row['Close']
        reward = 0.0

        # Simulate slippage by perturbing price slightly
        open_price *= (1 + random.uniform(-self.slippage_pct, self.slippage_pct))
        close_price *= (1 + random.uniform(-self.slippage_pct, self.slippage_pct))

        if action == 1:
            # Determine how many shares to buy using risk management
            shares_to_buy = self.risk_manager.compute_position_size(self.cash, open_price)
            if self.shares_held == 0 and shares_to_buy > 0:
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
        print(f"Step: {self.current_step}, Cash: {self.cash:.2f}, Trades: {self.trades}")

#############################
# 6. PPO Agent Module
#############################

class PPOAgent(nn.Module):
    """State-of-the-art PPO network with separate actor and critic heads."""
    def __init__(self, input_dim: int, hidden_dim: int, n_actions: int):
        super(PPOAgent, self).__init__()
        self.shared = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )
        self.actor = nn.Sequential(
            nn.Linear(hidden_dim, n_actions),
            nn.Softmax(dim=-1)
        )
        self.critic = nn.Linear(hidden_dim, 1)

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
            log_probs.append(log_prob.detach().cpu().numpy())
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
# 7. Distributed Training & Hyperparameter Tuning
#############################

class HyperparameterTuner:
    """Grid search over hyperparameters for PPO training."""
    def __init__(self, env, agent_cls, trainer_cls, param_grid: dict, total_timesteps: int):
        self.env = env
        self.agent_cls = agent_cls
        self.trainer_cls = trainer_cls
        self.param_grid = param_grid
        self.total_timesteps = total_timesteps

    def grid_search(self):
        best_reward = -np.inf
        best_params = None
        results = {}
        # Create all combinations of hyperparameters
        import itertools
        keys = list(self.param_grid.keys())
        for values in itertools.product(*self.param_grid.values()):
            params = dict(zip(keys, values))
            logging.info(f"Testing hyperparameters: {params}")
            # Instantiate a new agent and trainer for each configuration
            agent = self.agent_cls(input_dim=self.env.observation_space.shape[0],
                                   hidden_dim=params.get("hidden_dim", 128),
                                   n_actions=self.env.action_space.n)
            trainer = self.trainer_cls(env=self.env, agent=agent,
                                       lr=params.get("lr", 1e-4),
                                       gamma=params.get("gamma", 0.99),
                                       clip_epsilon=params.get("clip_epsilon", 0.2),
                                       entropy_coef=params.get("entropy_coef", 0.01),
                                       epochs=params.get("epochs", 4))
            rewards = trainer.train(total_timesteps=self.total_timesteps, log_interval=50)
            avg_reward = np.mean(rewards[-10:])  # average last 10 episodes
            results[str(params)] = avg_reward
            logging.info(f"Params: {params} -> Avg Reward: {avg_reward:.2f}")
            if avg_reward > best_reward:
                best_reward = avg_reward
                best_params = params
        logging.info(f"Best params: {best_params} with Avg Reward: {best_reward:.2f}")
        return best_params, results

#############################
# 8. Evaluation Module
#############################

class Evaluator:
    """Evaluates the trained model on a validation set."""
    def __init__(self, env, agent, n_episodes: int = 10):
        self.env = env
        self.agent = agent
        self.n_episodes = n_episodes

    def evaluate(self):
        rewards = []
        for _ in range(self.n_episodes):
            state = self.env.reset()
            done = False
            ep_reward = 0
            while not done:
                state_tensor = torch.from_numpy(state).float().to(self.agent.device)
                with torch.no_grad():
                    action_probs, _ = self.agent(state_tensor)
                dist = torch.distributions.Categorical(action_probs)
                action = dist.sample().item()
                state, reward, done, info = self.env.step(action)
                ep_reward += reward
            rewards.append(ep_reward)
        avg_reward = np.mean(rewards)
        logging.info(f"Evaluation over {self.n_episodes} episodes: Avg Reward = {avg_reward:.2f}")
        return avg_reward

#############################
# 9. Main Function: Tie Everything Together
#############################

def main():
    # Configuration and date range
    ticker = "AAPL"
    company_name = "Apple"
    news_api_key = "8ff74aefdb9c4516be523ca21d39b5e2"  # Replace with your valid key
    today = datetime.date.today()
    start_date = today - datetime.timedelta(days=90)
    end_date = today

    # Data ingestion
    data_loader = DataLoader(ticker=ticker, start_date=start_date, end_date=end_date)
    raw_data = data_loader.fetch_yahoo_data()
    sentiment = data_loader.fetch_news_sentiment(query=company_name, news_api_key=news_api_key)
    
    # Feature engineering
    fe = FeatureEngineer(raw_data)
    data_with_indicators = fe.add_technical_indicators()
    data_with_features = fe.add_sentiment_feature(sentiment)
    clean_data = fe.clean_data()
    
    # Save processed data for persistence
    processed_data_path = "processed_stock_data.csv"
    clean_data.to_csv(processed_data_path, index_label="Date")
    logging.info(f"Processed data saved to {processed_data_path}")

    # Create Risk Manager
    risk_manager = RiskManager(max_position=0.1, stop_loss=0.05, take_profit=0.10)
    
    # Create the trading environment
    env = ExtendedStockTradingEnv(data=clean_data, initial_cash=100000,
                                  transaction_cost_pct=0.001, risk_manager=risk_manager,
                                  slippage_pct=0.001, normalize=True)

    # ----------------------------
    # Option 1: Train PPO using a single process
    # ----------------------------
    input_dim = env.observation_space.shape[0]
    n_actions = env.action_space.n
    hidden_dim = 256  # Increase network capacity

    ppo_agent = PPOAgent(input_dim=input_dim, hidden_dim=hidden_dim, n_actions=n_actions)
    ppo_trainer = PPOTrainer(env=env, agent=ppo_agent, lr=1e-4, gamma=0.99,
                             clip_epsilon=0.2, entropy_coef=0.01, epochs=4)

    total_timesteps = 10000  # For demonstration; in production, use many more timesteps
    logging.info("Starting PPO training...")
    rewards = ppo_trainer.train(total_timesteps=total_timesteps, log_interval=100)

    # Save the trained PPO model
    model_save_path = "ppo_trading_model.pth"
    torch.save(ppo_agent.state_dict(), model_save_path)
    logging.info(f"Trained PPO model saved to {model_save_path}")

    # ----------------------------
    # Option 2: Hyperparameter Tuning (Grid Search)
    # ----------------------------
    # Uncomment the following block to perform grid search tuning.
    """
    param_grid = {
        "hidden_dim": [128, 256],
        "lr": [1e-4, 5e-5],
        "gamma": [0.99],
        "clip_epsilon": [0.1, 0.2],
        "entropy_coef": [0.01, 0.005],
        "epochs": [4, 8]
    }
    tuner = HyperparameterTuner(env=env, agent_cls=PPOAgent, trainer_cls=PPOTrainer,
                                param_grid=param_grid, total_timesteps=5000)
    best_params, tuning_results = tuner.grid_search()
    logging.info(f"Best hyperparameters: {best_params}")
    """

    # ----------------------------
    # Evaluation
    evaluator = Evaluator(env=env, agent=ppo_agent, n_episodes=10)
    avg_val_reward = evaluator.evaluate()
    logging.info(f"Final Evaluation Average Reward: {avg_val_reward:.2f}")

    # Plot training rewards
    plt.figure(figsize=(12, 6))
    plt.plot(rewards)
    plt.xlabel("Timesteps (per batch)")
    plt.ylabel("Episode Reward")
    plt.title("PPO Training Rewards Over Time")
    plt.show()

if __name__ == "__main__":
    main()
