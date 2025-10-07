import ffn


prices = ffn.get('aapl', start='2010-01-01')

ax = prices.rebase().plot()
