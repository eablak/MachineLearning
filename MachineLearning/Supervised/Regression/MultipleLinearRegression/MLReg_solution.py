# -*- coding: utf-8 -*-
"""
Created on Fri Jan  5 09:37:02 2024

@author: esraablak
"""

import pandas as pd
import numpy as np


veriler = pd.read_csv("odev_tenis.csv")

# encoder
"Label encoding kategorik değişkenler arasında bir sıralama veya derecelendirme varsa, yani kategoriler arasında belirli bir anlam ilişkisi varsa, label encoding kullanılabilir. Örneğin düşük, orta, yüksek gibi sıralanabilir kategorilerin olduğu durumlarda."

"burada windy ve play 2 türden oluşuyor onehotencoding yaparsak dummy variable tuzağına düşeriz o yüzden label encoding yeterli olur. ama outlook için 3 değer var ona labelencoding yapamayız çünkü aralarında sıralama ilişkisi yok"

# DATA PREPROCESSING

"********************benim çözümüm*********************************************"

from sklearn.preprocessing import OneHotEncoder

ohe = OneHotEncoder()

outlook = veriler.iloc[:,0:1].values
outlook = ohe.fit_transform(outlook).toarray()

from sklearn.preprocessing import LabelEncoder

# 1) int olarak belirt ve dönüşüm yap
#windy = veriler.iloc[:,3:4].values.astype(int)

# 2) labelencoder ile 
le = LabelEncoder()
windy = veriler.iloc[:,3:4].values
windy = le.fit_transform(windy)

play = veriler.iloc[:,-1:].values
play[:,0] = le.fit_transform(play[:,0])

# windy true false olduğu için bu şekilde değişmiyor çünkü zaten arka planda numeric.


"*******************orijinal***************************************************"

# tüm kolonlara tek seferde labelencoder yapar. dikkat numeric verilere de yapar!
veriler2 = veriler.apply(LabelEncoder().fit_transform)

c = veriler2.iloc[:,:1]
ohe = OneHotEncoder()
c = ohe.fit_transform(c).toarray()

"******************************************************************************"

outlook_df = pd.DataFrame(data=c, index=range(14), columns=["o","r","s"])

sonveriler = pd.concat([outlook_df,veriler.iloc[:,1:3]], axis=1)
sonveriler = pd.concat([veriler2.iloc[:,-2:],sonveriler], axis=1)


# split !

from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(sonveriler.iloc[:,:-1], sonveriler.iloc[:,-1:],
                                                    test_size=0.33, random_state=0)


# linear regression!

from sklearn.linear_model import LinearRegression

regression = LinearRegression()

regression.fit(x_train,y_train)
y_pred = regression.predict(x_test)

# backwardElimination (iyileştirme / p-value hesaplama)

import statsmodels.api as sm

X = np.append(arr=np.ones((14,1)).astype(int), values=sonveriler.iloc[:,:-1], axis=1)
X_l = sonveriler.iloc[:,[0,1,2,3,4,5]].values
r_ols = sm.OLS(endog=sonveriler.iloc[:,-1], exog=X_l) # endog bağımlı, exog bağımsız olan veirler
r = r_ols.fit()
print(r.summary())

"çıktıya göre 0. kolonun p-value'si yüksek onu at"

sonveriler = sonveriler.iloc[:,1:]

X = np.append(arr=np.ones((14,1)).astype(int), values=sonveriler.iloc[:,:-1], axis=1)
X_l = sonveriler.iloc[:,[0,1,2,3,4]].values
r_ols = sm.OLS(endog=sonveriler.iloc[:,-1], exog=X_l)
r = r_ols.fit()
print(r.summary())

"0. kolonu atınca değerler iyileşti o zaman x_train ve x_testte de at ve modelini iyileştir yani tekrar train et"

x_train = x_train.iloc[:,1:]
x_test = x_test.iloc[:,1:]

regression.fit(x_train,y_train)
y_pred = regression.predict(x_test)

"y_pred ve y_test karşılaştırmasında modelin şu an iyi seviyede olduğu görülüyor"


