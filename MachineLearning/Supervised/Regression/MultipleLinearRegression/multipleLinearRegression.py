# -*- coding: utf-8 -*-
"""
Created on Thu Jan  4 15:01:02 2024

@author: esraablak
"""

"""
y = B0 + B1x1 + B2x2 + B3x3 + e.

!! Kukla Değişken (Dummy Variable): Cinsiyet kolonu e ve k diye kategorik olarak doldurulmuştur ama sen bunu işleyebilmek için one hot encoding ile 0 ve 1'lere göre tekrar iki kolon oluşturarak tutmuşsundur totalde cinsiyet için birden fazla aynı anlama gelen kolonların olmuştur ve bu risk arz eder.Çünkü bu durum bazı algoritmalarda sonucu etkileyebiliyor. Dummy (kukla) veri ile orjinal veriyi aynı anda almamaya dikkat et.

!! p-value (olasılık değeri)

H0: null hypothesis: Farksızlık hipotezi, sıfır hipotezi, boş hipotez. Kurabiye üretiyosun ve paketlenen kutuların hepsinde 100 adet kurabiye olduğunu söylüyosun. Bu tüm kutular 100 kurabiye içerir demek yani H0 ortaya atıyorsun. Veya ders çalışma süresi arttıkça ders başarısı artar gibi herhangi bir hipotez. İlk başta kabul ettiğimiz hipotez.

H1: Alternatif hipotez. H0 tersi. Ders çalışma süresi arttıkça başarı artmaz veya her zaman artmaz. Her kutuda 100 kurabiye yoktur gibi.

p-value tam bu ikisi arasındaki ilişki kaç tane h1 için kanıt bulursam h0'ı reddedip h1'i kabul edicem? 1-2 tane 90 adet kurabiye içeren kutu buldun ama 100 tane 100 adetlik kurabiye kutun var o zaman h0 mı h1'mi? sonuçta h1 için kanıt buldun. Bunun cevabı p-value'ya bağlı.

p-value: olasılık değeri (genellikle 0.05)
p-değeri küçüldükçe H0 hatalı olma ihtimali artar.
p-değeri büyüdükçe H1 hatalı olma ihtimali artar.


!! Çok Değişkenli Modellerde, Değişken Seçimi
a) Bütün değişkenleri dahil etmek
b) Geriye doğru eleme (Backward elimination)
c) İleri seçim (Forward selection)
d) İki yönlü eleme (bidirectional elimination)
e) Skor karşılaştırması (Score Comparison)

b-c-d => Adım adım karşılaştırma (stepwise)

Geriye Eleme (Backward Elimination)

1) Significance Level (SL) seçilir (genelde 0.05)
2) Bütün değişkenler kullanılarak bir model inşa edilir.
3) En yükse p-value değerine sahip olan değişken ele alınır ve şayet P>SL ise 4.adıma, değilse son adıma (6. adım) gidilir.
4) Bu aşamada, 3. adımda seçilen ve en yüksek p-değerine sahip değişken sistemden kaldırılır.
5) Makine öğrenmesi güncellenir ve 3. adıma geri dönülür.
6) Makine öğrenmesi sonlandırılır.


İleriye Seçim (Foward Selection)

1) Significance Level (SL) seçilir (genelde 0.05)
2) Bütün değişkenler kullanılarak bir model inşa edilir.
3) En düşük p-value değerine sahip olan değişken ele alınır
4) Bu aşamada, 3. adımda seçilen değişken sabit tutularak yeni bir değişken daha seçilir vve sisteme eklenir
5) Makine öğrenmesi güncellenir ve 3. adıma geri dönülür, şayet en düşük p-değere sahip değişken için p<SL şartı sağlanıyorsa 3.adıma dönülür. sağlanmıyorsa biter (6.adıma geçilir)
6) Makine öğrenmesi sonlandırılır.


Çift Yönlü Eleme (Bidirectional Elimination)
1) Significance Level (SL) seçilir (genelde 0.05)
2) Bütün değişkenler kullanılarak bir model inşa edilir.
3) En düşük p-value değerine sahip olan değişken ele alınır
4) Bu aşamada, 3. adımda seçilen değişken sabit tutularak diğer bütün değişkenler sisteme dahil edilir ve en düşük p değerine sahip olan sistemde kalır.
5) SL değerinin altında olan değişkenler sistemde kalır ve eski değişkenlerden hiçbirisi sistemden çıkarılmaz
6) Makine öğrenmesi sonlandırılır.

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
"""
sonuc3 için data=c[:,:1] yapmamızın sebebi dummy variable tuzağına düşmeyi engellemek için tek bir kolonu aldığımızda 0 veya 1 durumunda cinsiyet anlaşılır oluyor zaten 2 kolon almaya gerek yok
"""

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

"Aynı mantıkta bağımlı değişken olarak boyu atıyoruz kalan verilerle boy tahmini"

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





