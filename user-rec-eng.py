import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer

metadata = pd.read_csv('workdata.csv', sep = ',',error_bad_lines=False,
                       encoding = 'latin-1')
metadata.columns = ['work_id', 'title', 'synopsis', 'views']


tfidf = TfidfVectorizer(stop_words='english')
metadata['synopsis'] = metadata['synopsis'].fillna('')
mtrx = tfidf.fit_transform(metadata['synopsis'])
from sklearn.metrics.pairwise import linear_kernel
cosine = linear_kernel(mtrx, mtrx)


tfidf = TfidfVectorizer(stop_words='english')

metadata['synopsis'] =  metadata['synopsis'].fillna('')

mtrx = tfidf.fit_transform(metadata['synopsis'])

from sklearn.metrics.pairwise import linear_kernel
cosine = linear_kernel(mtrx, mtrx)


user_data = pd.read_csv('testdata.csv', sep = ',',error_bad_lines=False,
                   encoding = 'latin-1')
user_data.columns = ['reader_id', 'position', 'work_id', 'title', 'synopsis']



def get_recommendations(itemlist, cosine=cosine):
    recomm = []
    for x in itemlist:
        recomm.append(metadata.index[metadata['work_id']==x][0]) #make recomm a matrix with
        #the indices of the movies listed
    
    recom = [] #array for tuples of recommendations
    for x in recomm:
        sim_scores = list(enumerate(cosine[x]))
#         sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
        for y in sim_scores:
            val = y[1]*((user_data['position'].iloc[x]+1)/100) #weigh movi based off of position
            mytuple = (y[0], val)
            if y[0] not in recomm:
                recom.append(mytuple) #add tuple 


    recom = sorted(recom, key=lambda x: x[1], reverse=True) 
    recom = recom[1:9]

    # Get the movie indices
    movie_indices = [i[0] for i in recom]

    # Return the top 7 most similar movies
    return metadata['title'].iloc[movie_indices]


item_list = [14, 18, 22, 31, 43, 74]
get_recommendations(item_list)