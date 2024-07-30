# test_df.py
# test  data load
import numpy as np
import pandas as pd

df = pd.read_csv('./data/test_scores.csv')
x = df['math'].to_numpy()
y = df['cs'].to_numpy()

print(x)
print(y)
