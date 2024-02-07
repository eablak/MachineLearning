# -*- coding: utf-8 -*-
"""
Created on Thu Jan  4 15:01:02 2024

@author: esraablak
"""


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

veriler = pd.read_csv("veriler.csv")

# eksik veriler
from sklearn.impute import SimpleImputer
imputer = SimpleImputer(missing_values=np.NaN, strategy="mean")

Yas = veriler.iloc[:,1:4].values

imputer = imputer.fit(Yas[:,1:4])
Yas[:,1:4] = imputer.transform(Yas[:,1:4])


# encoder
ulke = veriler.iloc[:,0:1].values
from sklearn import preprocessing
le = preprocessing.LabelEncoder()
ulke[:,0] = le.fit_transform(ulke[:,0])


ohe = preprocessing.OneHotEncoder(categories="auto")
ulke = ohe.fit_transform(ulke).toarray()

c = veriler.iloc[:,-1:].values
c[:,-1] = le.fit_transform(veriler.iloc[:,-1])
c = ohe.fit_transform(c).toarray()

sonuc = pd.DataFrame(data=ulke, index=range(22), columns=["fr","tr","us"])
sonuc2 = pd.DataFrame(data=Yas, index=range(22), columns=["boy","kilo","yas"])
#cinsiyet = veriler.iloc[:,-1].values
sonuc3 = pd.DataFrame(data=c[:,:1], index=range(22), columns=["cinsiyet"])

s=pd.concat([sonuc,sonuc2],axis=1)
s2=pd.concat([s,sonuc3],axis=1)



from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(s,sonuc3,
                                    test_size=0.33, random_state=0)

from sklearn.preprocessing import StandardScaler

sc = StandardScaler()
X_train = sc.fit_transform(x_train)
X_test = sc.transform(x_test)


from sklearn.linear_model import LinearRegression

regression = LinearRegression()

regression.fit(x_train,y_train)
y_pred = regression.predict(x_test)


boy = s2.iloc[:,3:4].values
print(boy)

sol = s2.iloc[:,:3]
sag = s2.iloc[:,4:]

veri = pd.concat([sol,sag],axis=1)

x_train, x_test, y_train, y_test = train_test_split(veri,boy,
                                    test_size=0.33, random_state=0)

r2 = LinearRegression()
r2.fit(x_train,y_train)
y_pred = r2.predict(x_test)


"******************************************************************************"

import statsmodels.api as sm

X = np.append(arr=np.ones((22,1)).astype(int), values=veri, axis=1) # sabit terim oluşturuyoruz
"Sabit terimi (intercept) eklemek, genellikle regresyon modellerinde kullanılan bir tekniktir. Bağımsız değişkenlerle ilişkili olan katsayılar, bir regresyon denklemi içinde bu sabit terimi göz önünde bulundurarak tahmin edilir. Özellikle doğrusal regresyon modellerinde, bir doğrunun y-eksenini kesitği noktayı (yani ,bağımsız değişkenlerin değeri sıfır olduğunda bağımlı değişken değeri) temsil eder. Bu, veri setindeki değişkenlerin değerleriyle birlikte modelin bağımlı değişkeni nasıl etiketlediğini anlamak için önemlidir."

X_l = veri.iloc[:,[0,1,2,3,4,5]].values
X_l = np.array(X_l,dtype=float)
model = sm.OLS(boy,X_l).fit()
print(model.summary())

"P değerlerine baktığımızda 4. kolonun p değeri yüksek çıktığı için eliyoruz"

X_l = veri.iloc[:,[0,1,2,3,5]].values
X_l = np.array(X_l,dtype=float)
model = sm.OLS(boy,X_l).fit()
print(model.summary())















