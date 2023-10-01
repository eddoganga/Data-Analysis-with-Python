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
cols = ['MovieLense', 'Cast', 'Crew']
for col in cols:
   dummies.append(pd.get_dummies(df[col]))

data.columns=['movie_title', 'genre', 'release date', 'director', 'actor'] data.drop('actor',axis=1,inplace=True)



```
#Feature Engineering
```
from surprise import SVD
import numpy as np
import surprise
from surprise import Reader, Dataset
# It is to specify how to read the data frame.
reader = Reader(rating_scale=(1,5))
# create the traindata from the data frame
train_data_mf = Dataset.load_from_df(train_data[['userId', 'movieId', 'rating']], reader)
# build the train set from traindata. 
#It is of dataset format from surprise library
trainset = train_data_mf.build_full_trainset()
svd = SVD(n_factors=100, biased=True, random_state=15, verbose=True)
svd.fit(trainset)
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




