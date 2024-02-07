# -*- coding: utf-8 -*-
"""
Created on Thu Jan 11 08:48:22 2024

@author: esraablak
"""

"""

            Predicted:     Predicted:
n=165           NO            YES

Actual:
  NO          TN=50          FP=10          60
  
Actual:
  YES         FN=5           TP=100         105
  
              55              110
           
              
False Positive (Tip1 Hata)
False Negative (Tip2 Hata)
            

Accuracy(doğruluk): Kaç doğru sınıflandırma var?
    - (TP+TN)/total = (100+50)/165 = 0.91

Misclassification Rate: Kaç yanlış sınıflandırma yapılmış?
    - (FP+FN)/total = (10+5)/165 = 0.09
    - 1 - Accuracy
    - "Error Rate"

True Positive Rate: Gerçekte Yes ise kaçı doğru sınıflandırılmış?
    - TP/actual yes = 100/105 = 0.95
    - "Sensitivity" veya "Recall" (duyarlılık)
    
False Positive Rate: Tahmin No ise bu sonuçların kaçı doğru?
    - FP/actual no = 10/60 = 0.17
    
Specificity: Gerçekte No ise bunların kaçı doğru sınıflanmış?
    - TN/actual no = 50/60 = 0.83
    - 1 - False Positive Rate
    
Precision(kesinlik): Tahmin Yes ise kaçı doğru?
    - TP / predicted yes = 100/110 = 0.91
    
Prevalence: Gerçekteki Yes dağılım oranı
    - Actual yes / total = 105/165 = 0.64
    
    
Eşik değer: Sınıflandırma problemleri için olasılık değerlerine göre ne şekilde bir sınıflandırma yapılması gerektiğinin bir kriteridir. Örn 0.5 eşik değerinde 0.5 ve üzeri tahminler 1 sınıfına 0.5 altında yapılan tahminler 0 sınfına ait olur. Eşik değerleri üzerinde oynamalar yaparak kesinlik ve duyarlılık gibi farklı metrikleri kendi isteklerimiz ve amaçlarımız doğrultusunda değiştirebiliriz.

Receiver Operating Characteristic (ROC)

Precision ve sensitivity ters ilişkiye sahiptir. Farklı eşik değerleri için biri artarken biri azalır. İlk olarak eşik değeri 0.5 ayarlarsan modelin tahminine göre karmaşıklık matrisinde kesinlik-duyarlılık ilişkisi bir türlü olur, eşik değeri 0.3 yaparsan başka türlü olur, 0.7 yaparsa başka olur gibi gibi.. Bütün eşik değerleri tek tek deneyemezsin. bunun için -> ROC

ROC için y tarafında duyarlılık x tarafında false positive rate çizgisi olan grafik vardır. Eşik değeri 0'dan başlatırsın ve arttırarak 1 e kadar gidersin. Çıkan tüm noktaları birleştirerek ROC eğrisini elde edersin. ROC eğrisini uygun eşik değeri seçebilmek için yorumlarsın.

Area Under the Curve (AUC)

AUC, ROC eğrisinin altında kalan alanı ifade eder. 0 ve 1 arasında değer alır ve 1'e ne kadar yakınsa model de bir o kadar başarılıdır. AUC değeri, aynı veri seti üzerinde eğitilmiş iki modelin başarısının karşılaştırılmasını kolaylaştırır.
    
    
    
    
    
    
    
"""