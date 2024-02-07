# -*- coding: utf-8 -*-
"""
Created on Mon Jan 15 08:27:34 2024

@author: esraablak
"""

"""
NLU(natural language understanding)
NLG(natural language generation)

- Linguistik (dilbilim) yaklaşımı
    * Pragmatics (kullanımbilim)
    * Semantics (anlambilim)
    * syntax (sözdizim)
    * morphology (şekilbilim)
- İstatistiksel yaklaşım
    * N-gram
    * TF-IDF
    * Word-Gram
    * BOW (bag of words)
- Hibrit yaklaşımlar
"""

import numpy as np
import pandas as pd


yorumlar = pd.read_csv('Restaurant_Reviews.csv', on_bad_lines="skip")

# ! Regular Expression

import re
#substitute
yorum = re.sub('[^a-zA-Z]',' ', yorumlar['Review'][0])

yorum = yorum.lower()
yorum = yorum.split()

# stop words

import nltk

nltk.download("stopwords")
from nltk.corpus import stopwords

# stemmer (kelime gövdeleri /kelimeyi ek ve köklere ayırma)
from nltk.stem.porter import PorterStemmer
ps = PorterStemmer()

yorum = [ps.stem(kelime) for kelime in yorum if not kelime in set(stopwords.words("english"))]

yorum = ' '.join(yorum)

# Preprocessing
derlem = []
for i in range(len(yorumlar)):
    yorum = re.sub('[^a-zA-Z]',' ', yorumlar["Review"][i])
    yorum = yorum.lower()
    yorum = yorum.split()
    yorum = [ps.stem(kelime) for kelime in yorum if not kelime in set(stopwords.words("english"))]
    yorum = ' '.join(yorum)
    derlem.append(yorum)    
    
# Feature Extraction
    # Bag of Words
from sklearn.feature_extraction.text import CountVectorizer

cv = CountVectorizer(max_features=2000)
X = cv.fit_transform(derlem).toarray()    
y = yorumlar.iloc[:,1].values

# Machine Learning    
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(X,y, test_size=0.20, random_state=0)

eksik_veriler_train=np.isnan(y_train)
y_train[eksik_veriler_train]=0
eksik_veriler_test=np.isnan(y_test)
y_test[eksik_veriler_test]=0


from sklearn.naive_bayes import GaussianNB

gnb = GaussianNB()
gnb.fit(x_train,y_train)
y_pred = gnb.predict(x_test)

from sklearn.metrics import confusion_matrix

cm = confusion_matrix(y_test,y_pred)
print(cm)