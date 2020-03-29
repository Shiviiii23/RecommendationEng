import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

books = pd.read_csv('prol-test2.csv', sep = ',', error_bad_lines = False,
                   encoding = 'latin-1')
books.columns = ['user_id', 'work_title', 'work_rating']



df = pd.DataFrame(
    books.groupby(['work_title', 'user_id']).mean().unstack())
new = pd.DataFrame(df.fillna(0))
print(new)

#make 2-D matrix based off of this:
from sklearn.neighbors import NearestNeighbors


model = NearestNeighbors(metric = 'cosine', algorithm = 'brute')
model.fit(new)

idx = np.random.choice(new.shape[0])

distance, indices = model.kneighbors(new.iloc[idx, :].values.reshape(1, -1), n_neighbors = 6)

for i in range(0, len(distance.flatten())):
    if i == 0:
        print('Recommendation for: {0}'.format(new.index[idx]))
    else:
        print('{0}: {1}'.format(i, new.index[indices.flatten()[i]]))