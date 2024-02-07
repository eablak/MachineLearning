# -*- coding: utf-8 -*-
"""
Created on Tue Jan 16 09:36:03 2024

@author: esraablak
"""

"""
Linear Discriminant Analysis

LDA, genel olarak PCA'a benzesede LDA'in çalışma mantığında sınıflar arasındaki uzaklığı maksimize etmek vardır. PCA'da sınıf kavramı yoktur. PCA sadece data pointler arası mesaeyi maksimize etmeye çalışır.
LDA amacı sınıflar arasındaki farkı maksimize ederek veri setinde boyut indirgemektir.

* PCA benzeri bir boyut dönüştürme/indirgeme algoritmasıdır.
* PCA'den farklı olarak sınıflar arasındaki ayrımı önemser ve maksimize etmeye çalışır.
* PCA bu açıdan -> unsupervised LDA ise -> supervised özelliktedir.
"""

import pandas as pd

veriler = pd.read_csv("Wine.csv")

X = veriler.iloc[:,0:13].values
y = veriler.iloc[:,13].values

from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(X,y, test_size=0.2, random_state=0)

from sklearn.preprocessing import StandardScaler

sc = StandardScaler()

x_train = sc.fit_transform(x_train)
x_test = sc.transform(x_test)


from sklearn.decomposition import PCA

pca = PCA(n_components=2)
x_train2 = pca.fit_transform(x_train)
x_test2 = pca.transform(x_test)

from sklearn.linear_model import LogisticRegression

# pca dönüşümünden önce LR
classifier = LogisticRegression(random_state=0)
classifier.fit(x_train,y_train)

# pca dönüşümünden sonra LR
classifier2 = LogisticRegression(random_state=0)
classifier2.fit(x_train2,y_train)

y_pred = classifier.predict(x_test)
y_pred2 = classifier2.predict(x_test2)

from sklearn.metrics import confusion_matrix

cm = confusion_matrix(y_test,y_pred)
print("PCA'siz")
print(cm)

cm2 = confusion_matrix(y_test, y_pred2)
print("PCA'lı")
print(cm2)

cm3 = confusion_matrix(y_pred,y_pred2)
print("PCA'siz ve PCA'li")


"*********************************************************************************************************************"

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA

lda = LDA(n_components=2)

x_train_lda = lda.fit_transform(x_train,y_train)
" lda için fit transform yaparken hem x_train hem y_train verirsin çünkü sınıflar arası mesafeyi maksimize etmek istersin ama pca için sadece xtrain !!"
x_test_lda = lda.transform(x_test)

classifier_lda = LogisticRegression(random_state=0)
classifier_lda.fit(x_train_lda,y_train)

y_pred_lda = classifier_lda.predict(x_test_lda)

print("lda ve orijinal")
cm_lda = confusion_matrix(y_pred,y_pred_lda)
print(cm_lda)

" T-distributed Stochastic Neighbor Embedding (t-SNE), özellikle yüksek boyutlu veri setlerini görselleştirmek amacıyla kullanılan bir makine öğrenimi algoritmasıdır. t-SNE, genellikle görselleştirme amacıyla kullanıldığından, öğrenilen düşük boyutlu gömme, orijinal veri setindeki örneklerin yapısını daha iyi yansıtan bir uzayda yer almaya eğilimlidir"