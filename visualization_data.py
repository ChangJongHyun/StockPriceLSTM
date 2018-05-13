import datetime
import matplotlib.pyplot as plt
import seaborn
from tools import fetch_stock_price, format_dataset
from tools import fetch_cosine_values

cos_values = fetch_cosine_values(20, frequency=0.1)
seaborn.tsplot(cos_values)

plt.xlabel("Days since start of the experiment")
plt.ylabel("Value of the cosine function")
plt.title("Cosine time series over time")
plt.show()

feature_size = 5
minibatch_cos_x, minibatch_cos_y = format_dataset(cos_values, feature_size)
print("minibatch_cos_x.shape=", minibatch_cos_x.shape)
print("minibatch_cos_y.shape=", minibatch_cos_y.shape)

# page 117 안되네 subplot parameter 문제 같음
sample_to_plot = 5
# -------------------------------------------

symbols = ["MSFT", "FB", "SWK", "AAPL"]
ax = plt.subplot(1, 1, 1)
for sym in symbols:
    prices = fetch_stock_price(sym, datetime.date(2015, 1, 1), datetime.date(2017, 12, 13))
    ax.plot(range(len(prices)), prices, label=sym)

handles, labels = ax.get_legend_handles_labels()
ax.legend(handles, labels)
plt.xlabel("Trading days since 2015-1-1")
plt.ylabel("Stock price {$}")
plt.title("Prices of some American stocks in trading days of 2015 and 2016")
plt.show()
