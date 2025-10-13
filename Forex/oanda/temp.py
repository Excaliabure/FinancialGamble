import torch
import torch.nn as nn

pairs = [
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


network = nn.Sequential(
            nn.Linear(len(pairs), 128),
            nn.ReLU(),
            nn.Linear(128, 54),
            nn.ReLU(),
            nn.Linear(54,128),
            nn.Sigmoid()  # Each element is probability of "buy"
        )

a = torch.randn((len(pairs), 500)).numpy()

# print(network(a.T).shape)
print(type(a))