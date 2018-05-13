import os
import pickle

import quandl as quandl
import datetime
import numpy as np


# pretend random cosine values as stock price
def fetch_cosine_values(seq_len, frequency=0.01, noise=0.1):
    np.random.seed(1010)
    x = np.arange(0.0, seq_len, 1.0)
    return np.cos(2 * np.pi * frequency * x) + np.random.uniform(low=-noise,
                                                                 high=noise, size=seq_len)


def date_obj_to_str(date_obj):
    return date_obj.strftime('%Y-%m-%d')


def save_pickle(something, path):
    if not os.path.exists(os.path.dirname(path)):
        os.makedirs(os.path.dirname(path))
    with open(path, 'wb') as fh:
        pickle.dump(something, fh, pickle.DEFAULT_PROTOCOL)


def load_pickle(path):
    with open(path, 'rb') as fh:
        return pickle.load(fh)


def fetch_stock_price(symbol, from_date, to_date,
                      cache_path="./tmp/prices/"):
    assert (from_date <= to_date)
    filename = "{}_{}_{}.pk".format(symbol, str(from_date), str(to_date))
    price_filepath = os.path.join(cache_path, filename)
    try:
        prices = load_pickle(price_filepath)
        print("loaded from", price_filepath)
    except IOError:
        historic = quandl.get("WIKI/" + symbol, start_date=date_obj_to_str(from_date),
                              end_date=date_obj_to_str(to_date))
        prices = historic["Adj. Close"].tolist()
        save_pickle(prices, price_filepath)
        print("saved into ", price_filepath)
    return prices


print(fetch_stock_price("GOOG", datetime.date(2017, 1, 1),
                        datetime.date(2017, 1, 31)))


# 미리 정해둔 크기만 사용 --> but 다양한 길이를 사용하고 싶어

def format_dataset(values, temporal_features):
    feat_splits = [values[i:i + temporal_features] for i in
                   range(len(values) - temporal_features)]
    feats = np.vstack(feat_splits)
    labels = np.array(values[temporal_features:])
    return feats, labels


# matrix to array
def matrix_to_array(m):
    return np.asarray(m).reshape(-1)