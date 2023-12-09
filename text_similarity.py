#!/usr/bin/env python
# coding: utf-8

# In[3]:


from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd
import spacy


# In[4]:


nlp = spacy.load("en_core_sci_scibert")


# In[5]:


def process_text(text): 
    doc = nlp(text)
    return " ".join([ent.text for ent in doc.ents])


# In[17]:


def calculate_similarity(main_text, other_text):
    processed_main_text = process_text(main_text)
    processed_other_text = process_text(other_text)
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform([processed_main_text, processed_other_text])
    cosine_similarities = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:]).flatten()
    return cosine_similarities[0]

