# region imports
from AlgorithmImports import *
from datetime import timedelta
import numpy as np
import torch
import torch.nn as nn
import pandas as pd
# endregion

import torch.nn.functional as F

class PPOAgent(nn.Module):
    """Conv1D PPO agent for time-series forex trading."""
    def __init__(self, num_channels=4, seq_len=60, num_pairs=54, hidden_dim=64):
        super().__init__()
        self.num_pairs = num_pairs
        self.seq_len = seq_len

        # Conv1D to learn temporal features per pair
        self.conv = nn.Sequential(
            nn.Conv1d(in_channels=num_channels, out_channels=16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1)  # collapse time dimension
        )

        # Fully connected layers after convolution
        self.fc = nn.Sequential(
            nn.Linear(64, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),  # output probability per pair
        )

    def forward(self, x):
        """
        x: tensor of shape (num_pairs, channels, seq_len)
        returns: tensor of shape (num_pairs,) -> buy probabilities
        """
        outputs = []
        for i in range(x.shape[0]):  # loop over actual number of pairs
            xi = x[i].unsqueeze(0)          # shape (1, channels, seq_len)
            conv_out = self.conv(xi)        # shape (1, 32, 1)
            conv_out = conv_out.view(1, -1) # flatten: (1, 32)
            prob = self.fc(conv_out)        # shape (1, 1)
            outputs.append(prob)
        return torch.cat(outputs, dim=0).squeeze()  # shape (num_pairs,)




class SquareFluorescentPinkLemur(QCAlgorithm):
    def Initialize(self):
        self.SetStartDate(2017, 4, 11)
        self.SetEndDate(2024, 4, 11)
        self.SetCash(1_000_000)

        # Forex universe
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
        self.symbols = [self.AddForex(p, Resolution.MINUTE, Market.Oanda).Symbol for p in self.pairs]

        # PPO agent setup
        self.agent_actor = PPOAgent(num_channels=4, seq_len=60, num_pairs=len(self.symbols))
        self.agent_critic = PPOAgent(num_channels=4, seq_len=60, num_pairs=len(self.symbols))

        self.optimizer_actor = torch.optim.Adam(self.agent_actor.parameters(), lr=1e-3)
        self.optimizer_critic = torch.optim.Adam(self.agent_critic.parameters(), lr=1e-3)

        # Buffers
        self.hist = []
        self.training_interval = 24  # daily
        self.prev_prices = np.zeros(len(self.symbols))
        self.curr_prices = np.zeros(len(self.symbols))
        self.inital_iters = 9999
        self.reward_arr = [1]

        # Schedules
        self.Schedule.On(
            self.DateRules.EveryDay(),
            self.TimeRules.Every(timedelta(hours=1)),
            self.TradeRoutine
        )

    # Run TrainAgents every hour
        self.Schedule.On(
            self.DateRules.EveryDay(),
            self.TimeRules.Every(timedelta(hours=1)),
            self.TrainAgents
            )


        self.reward = np.zeros(len(self.symbols))
        self.Debug("Algorithm initialized.")

    def OnData(self, data: Slice):
        if self.inital_iters < 5:
            for i, sym in enumerate(self.symbols):
                if data.ContainsKey(sym):
                    self.curr_prices[i] = data[sym].Price
        else:
            self.inital_iters -= 1

 


    def TradeRoutine(self):

        # Calculate Reward
        for i in range(len(self.symbols)):
            self.reward[i] = self.portfolio[self.symbols[i]].UnrealizedProfit # profit/loss proportional to price change
            

        self.reward = np.clip(self.reward, -1, 1)

        
        if self.portfolio.invested:
            self.liquidate
        
        # Gets game state for the values of 
        state = self.GetGameState()
        state = torch.tensor(state, dtype=torch.float32).permute(1,0,2)
        
        # Decision direction of the market order
        logits_critic = self.agent_critic(state)
        slopes = F.tanh(logits_critic)

        # Machine Learning Layer of the decision
        actions = self.agent_actor(state)
        actions = F.sigmoid(actions)
        actions[actions>0.5] = 1

        

        # Buy Loop for all the pairs 
        for i, sym in enumerate(self.symbols):
            if actions[i] == 1 and slopes[i] != 0:
                self.market_order(sym, int(1_000 * slopes[i])) 

        


    def GetGameState(self, step_back=60):
        """
        Returns the recent market data as a NumPy array for the PPO agent.

        Shape: (channels, step_back, num_pairs)
        Channels order: [close, high, low, open]
        """
        num_pairs = len(self.symbols)
        channels = 4  # close, high, low, open
        end = self.Time
        start = end - timedelta(minutes=step_back)

        # Retrieve history from QC
        history = self.History(self.symbols, start, end, Resolution.Minute)

        # Initialize state array with zeros
        state = np.zeros((channels, step_back, num_pairs))

        # Loop over each symbol to fill in available data
        for i, sym in enumerate(self.symbols):
            # Check if symbol has data in the history
            if sym in history.index.get_level_values(0):
                sym_hist = history.loc[sym].tail(step_back)
                # Fill available rows; missing data remains zero
                n = len(sym_hist)
                state[0, -n:, i] = sym_hist['close'].values
                state[1, -n:, i] = sym_hist['high'].values
                state[2, -n:, i] = sym_hist['low'].values
                state[3, -n:, i] = sym_hist['open'].values
            else:
                # No data for this symbol; leave zeros
                continue

        return np.array(state)



    def TrainAgents(self):
        if not self.hist:
            return

        # Convert reward to tensor
        reward = torch.tensor(self.reward, dtype=torch.float32)

        # Prepare state
        state = torch.tensor(self.hist, dtype=torch.float32).permute(1,0,2)  # shape: (batch, channels, pairs)

        # Zero gradients first
        self.optimizer_critic.zero_grad()
        self.optimizer_actor.zero_grad()

        # Forward pass
        value_pred_critic = self.agent_critic(state)  # shape: (num_pairs,)
        value_pred_actor = self.agent_actor(state)    # shape: (num_pairs,)

        # Backpropagate reward directly
        value_pred_critic.backward(gradient=reward)
        value_pred_actor.backward(gradient=reward)

        # Update parameters
        self.optimizer_critic.step()
        self.optimizer_actor.step()

        # Clear history and reward buffer
        self.hist = []
        self.reward_arr = []

        self.debug("Training Actor and Critic")
        self.debug(f"Current reward for both: {reward}")
