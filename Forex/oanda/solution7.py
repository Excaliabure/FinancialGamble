# region imports
from AlgorithmImports import *
from datetime import datetime, timedelta, time, date
import numpy as np
import torch
import torch.nn as nn
import pandas as pd
# endregion


class PPOAgent(nn.Module):
    """A simple PPO-style agent that outputs buy probabilities for each pair."""
    def __init__(self, state_dim, action_dim):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(12, 6),
            nn.ReLU(),
            nn.Linear(6, 1),
            nn.Sigmoid()  # Each element is probability of "buy"
        )

    def forward(self, x):
        return self.network(x)


class SquareFluorescentPinkLemur(QCAlgorithm):
    def initialize(self):
        self.set_start_date(2024, 4, 11)
        self.set_end_date(2024, 9, 11)
        self.set_cash(1_000_000)

        # Define forex pairs
        self.pairs = [
            'AUDCAD', 'AUDCHF', 'AUDHKD', 'AUDNZD', 'AUDSGD', 'AUDUSD',
            'CADCHF', 'CADHKD', 'CADSGD', 'CHFHKD', 'CHFZAR', 'EURAUD',
            'EURCAD', 'EURCHF', 'EURCZK', 'EURDKK', 'EURGBP', 'EURHKD',
            'EURNZD', 'EURPLN', 'EURSEK', 'EURSGD', 'EURTRY', 'EURUSD',
            'EURZAR', 'GBPAUD', 'GBPCAD', 'GBPCHF', 'GBPHKD', 'GBPNZD',
            'GBPPLN', 'GBPSGD', 'GBPUSD', 'GBPZAR', 'NZDCAD', 'NZDCHF',
            'NZDHKD', 'NZDSGD', 'NZDUSD', 'SGDCHF', 'USDCAD', 'USDCHF',
            'USDCNH', 'USDCZK', 'USDDKK', 'USDHKD', 'USDMXN', 'USDNOK',
            'USDPLN', 'USDSEK', 'USDSGD', 'USDTHB', 'USDTRY', 'USDZAR'
        ]

        # Add data
        self.symbols = [self.add_forex(p, Resolution.MINUTE, Market.OANDA).Symbol for p in self.pairs]

        # Schedule trading function (runs once per day)
        # self.schedule.on(
        #     self.date_rules.every_day(),
        #     self.time_rules.every(TimeSpan.from_hours(23)),
        #     self.TradePairs
        # )

        # Initialize PPO agent
        self.agent = PPOAgent(state_dim=len(self.pairs), action_dim=len(self.pairs))
        self.optimizer = torch.optim.Adam(self.agent.parameters(), lr=1e-3)

        # MIA KALIFA 
        self._symbols: list[Symbol] = []
        for ticker in self.pairs:
            symbol: Symbol = self.add_forex(ticker, Resolution.MINUTE).symbol
            self._symbols.append(symbol)

            self._previous_day_hourly_data: dict[Symbol, pd.DataFrame] = {}


        # State buffers
        self.prev_prices = np.zeros(len(self.symbols))
        self.curr_prices = np.zeros(len(self.symbols))
        self.hist = []  # stores (state, action, reward)
        self._state = []
        self.training_interval = 50  # train every 50 steps

        # self.debug("✅ PPO RL Agent Initialized")

    def on_data(self, data: Slice):
        """Collect state and train PPO periodically."""
        # 1️⃣ Build state vector
        updated = False
        for i, sym in enumerate(self.symbols):
            if data.contains_key(sym):
                self.curr_prices[i] = data[sym].Price
                updated = True

        if not updated:
            return

        # Skip first step
        if np.any(self.prev_prices == 0):
            self.prev_prices = self.curr_prices.copy()
            return

        # 2️⃣ Create state tensor
        hours = 12
        end = self.time
        start = end - timedelta(hours=hours)


       
        self._state, self.curr_prices = self.GetGameState(self.time)

        
        # state = torch.tensor(self._state, dtype=torch.float32)

        # 3️⃣ Get action probabilities (buy probability for each pair)
        with torch.no_grad():
            probs = self.agent(self._state)
            actions = (probs > 0.5).float()  # convert to binary {0,1}

        # 4️⃣ Execute trades in QuantConnect
        if self.time.hour % 23 == 0:
            for i, sym in enumerate(self.symbols):
                if actions[i] == 1:
                    self.market_order(sym, 1000)  # buy if action == 1
            

        # 5️⃣ Compute reward (PnL change)
        reward = np.array([self.Portfolio[sym].UnrealizedProfit for sym in self.symbols]).sum().item()

        # 6️⃣ Store transition
        self.hist.append((self._state, actions, reward))

        # 7️⃣ Train agent periodically
        if len(self.hist) >= self.training_interval:
            self.train_agent()
            self.hist = []  # clear memory

        # 8️⃣ Update prev prices
        self.prev_prices = self.curr_prices.copy()

    def train_agent(self):
        """Simple PPO-style policy gradient update."""
        states, actions, rewards = zip(*self.hist)
        states = torch.stack(states)
        actions = torch.stack(actions)
        rewards = torch.tensor(rewards, dtype=torch.float32)

        # Compute discounted returns
        gamma = 0.99
        returns = []
        G = 0
        for r in reversed(rewards):
            G = r + gamma * G
            returns.insert(0, G)
        returns = torch.tensor(returns, dtype=torch.float32)
        if self.time.hour == 12 and self.time.minute == 0:
            self.get_previous_day_hourly_data()
        # Normalize returns (stabilize training)
        returns = (returns - returns.mean()) / (returns.std() + 1e-8)

        # Forward pass through agent
        probs = self.agent(states)

        # PPO-like policy loss (maximize reward-weighted log-likelihood)
        log_probs = actions * torch.log(probs + 1e-8) + (1 - actions) * torch.log(1 - probs + 1e-8)
        loss = -(log_probs.sum(dim=1) * returns).mean()

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # self.debug(f"Trained Agent | Loss={loss.item():.6f}, Mean Reward={rewards.mean().item():.6f}")

   
    def GetGameState(self, dtime, step_back=12):
        start = dtime - timedelta(hours=step_back)
        end = dtime

        try:
            history = self.History(self.symbols, start, end, Resolution.HOUR)['close']

            if history.empty:
                raise ValueError("No historical data, using dummy fallback")
            
            history = history.unstack(level=0)  # columns = symbols, rows = hours
            history = history.fillna(method='ffill').fillna(1.0)

        except Exception:
            # Dummy fallback: ones
            history = pd.DataFrame(np.ones((step_back, len(self.symbols))),
                                columns=[s.Value for s in self.symbols])

        # Flatten into 1D vector for NN
        r = torch.tensor(history.to_numpy().T, dtype=torch.float32)
        curr = history.iloc[-1].to_numpy()

        return r, curr

