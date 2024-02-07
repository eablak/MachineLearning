# -*- coding: utf-8 -*-
"""
Created on Mon Jan 15 10:20:39 2024

@author: esraablak
"""

"""
Principal Component Analysis

Bir PCA analizinin ana amacı, değişkenler arasındaki korelasyonu tespit etmeyi amaçlamaktadır. Çok boyutlu verilere doğru açıdan bakarak genellikle verideki ilişkiler açıklanabilir. PCA'nın amacı bu "doğru açıyı" bulmaktır. PCA analizinin işleyiş sırası:
- Verilerin Boyutunu Azaltma
- Tahminleme Yapma
- Veriyi Görüntüleme
PCA: yüksek boyutlu verilerde maksimum varyansı bulmak ve bilgiyi korurken daha küçük boyutlara indirgemektir (verinin korunması %100 kalmaz)

EigenValue(öz-değer) - EigenVector(öz-vektör):

Eigen Vector => Bir yöneyin (vector) bir dönüşüme (transform) uğramasından sonra boyutunun değişmesinden bağımsız olarak hala yönü aynı kalıyorsa bu dönüşüm yöneyine (vector) öz yöney (eigen vector) denir.

Eigen Value => Bu yön değiştirmeyen ancak uzunluk (büyükllük) değiştiren öz yöneyin yapmış olduğu değişim aslında sayısal bir uzunluk olarak hesaplanabilir işte bu hesaplanan sayısal değere (sabit, scalar) öz değer  (eigen value) denir.

Bir kovaryans (veya korelasyon) matrisinin özvektörleri ve özdeğerleri, bir PCA'nın temelini oluşturur.
Özdeğerler, yeni özellik eksenleri boyunca verinin varyansını açıklar.


Boyut dönüştürme
Boyut indirgeme (gereksiz boyutlardan kurtulma veya bazı boyutları birleştirme)
Değişkenler arasındaki bağlantıları açığa çıkarma

PCA Algoritması

* İndirgenmek istenen boyut k olsun
* Veriyi standartlaştır
* Covariance(kovaryans) veya Corellation(korelasyın) matrisinden öz değerleri ve özvektörleri elde et. Veya SVD kullan.
* Öz değerleri büyükten küçüğe sırala ve k tanesini al
* Seçilen k özdeğerden W projeksyon matrisini oluştur
* Orijinal veri kümesi X'i W kullanarak dönüştür ve k-boyutlu Y uzayını elde et.
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
print(cm3)