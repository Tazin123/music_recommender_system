#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd


# In[4]:


df = pd.read_csv("spotify_millsongdata.csv")


# In[5]:


df.head(5)


# In[6]:


df.tail(5)


# In[7]:


df.shape


# In[8]:


df.isnull().sum()


# In[9]:


df =df.sample(5000).drop('link', axis=1).reset_index(drop=True)


# In[10]:


df.head(10)


# In[11]:


df['text'][0]


# In[12]:


df.shape


# Text Cleaning/ Text Preprocessing

# In[13]:


df['text'] = df['text'].str.lower().replace(r'^\w\s', ' ').replace(r'\n', ' ', regex = True)


# In[16]:


import nltk
nltk.download('punkt')
from nltk.stem.porter import PorterStemmer
stemmer = PorterStemmer()

def tokenization(txt):
    tokens = nltk.word_tokenize(txt)
    stemming = [stemmer.stem(w) for w in tokens]
    return " ".join(stemming)


# In[17]:


df['text'] = df['text'].apply(lambda x: tokenization(x))


# In[18]:


from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


# In[19]:


tfidvector = TfidfVectorizer(analyzer='word',stop_words='english')
matrix = tfidvector.fit_transform(df['text'])
similarity = cosine_similarity(matrix)


# In[20]:


similarity[0]


# In[21]:


df[df['song'] == 'Crying Over You']


# In[24]:


def recommendation(song_df):
    song_matches = df[df['song'] == song_df]
    if not song_matches.empty:
        idx = song_matches.index[0]
        distances = sorted(list(enumerate(similarity[idx])), reverse=True, key=lambda x: x[1])

        songs = []
        for i in distances[1:6]:  # Adjust the range as per your needs
            songs.append(df.iloc[i[0]]['song'])
        return songs
    else:
        return f"Song '{song_df}' not found in the dataset."


# In[27]:


recommendation('Deliver Your Children')


# In[28]:


import pickle
pickle.dump(similarity,open('similarity.pkl','wb'))
pickle.dump(df,open('df.pkl','wb'))


# In[ ]:




