# Data-Analysis-with-Python
#Data Collection
```
import pandas as pd
df = pd.read_csv('movies.csv')
```
#Data Preprocessing
```
df = pd.read_csv('movies.csv')
df.info()

df.dropna()

dummies = []
cols = ['Gender', 'Year of Release', 'Director']
for col in cols:
   dummies.append(pd.get_dummies(df[col]))
```
#Feature Engineering
```
# Removing percentage sign from RT critic score
for index, row in df.iterrows():
    if pd.notnull(row['rt_critic_score']):
        df.loc[index, 'rt_critic_score'] = int(row['rt_critic_score'][:2])
```
#Model selection
```
%matplotlib inline
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams['figure.figsize'] = 10, 7.5
plt.rcParams['axes.grid'] = True
plt.gray()
```




