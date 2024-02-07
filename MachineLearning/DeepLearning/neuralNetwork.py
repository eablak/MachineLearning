# -*- coding: utf-8 -*-
"""
Created on Sat Jan 13 10:09:04 2024

@author: esraablak
"""

"""
neural networkler normalize edilmiş şekilde çalışır (0-1 arasında).

Aktivasyon fonksiyonu: aslında en temelde nöronun(sinapslardan gelen girdilerle ağırlıkların çarpımının toplamından oluşan) sinyal ateşleyip ateşlemeyeceğine karar verir. Bir nöron nasıl cevap veriyor ? Cevap aktivasyon fonksiyonu. Yani sinapslardan gelen yükler toplanır ve aktivasyon fonksiyonu uygulanır ve çıkan sonuca göre karar veriyorlar (ateşlenip/ateşlenmeyeceği)
Hidden layer'lardaki her nöronun olayı diğer nöronlardan bağımsız olarak ateşlenip ateşlenmeyeceği kararını verir. Çıkış katmanı da (output layer) bir nöron aslında. sonuç yükleri çıkıştan okunur ve çıkış için de bağımsız başka bir aktivasyon fonksiyonu uygulanır. ve çıktı elde edilir.
Gizli katmanların için seçtiğin aktivasyon fonksiyonunu normalde tüm gizli katmanlarda kullanırsın. ama bu tüm katmanlar için aynı fonksiyon olmak zorunda değil istersen bunu değiştirebilrisin. çıkış için de yine akt. fonk seçer ve uygularsın.


Gizli katmanda bulunan bir nöron ile sonuç uzayını tek bir çizgi ile bölebilirsin ama iki nöron koyarsan (gizli katmandaki nöron sayısını ikiye çıkarırsan) iki çizgi elde edersin (bir nöron bir çizgiyi diğer nöron diğer çizgiyi temsil eder).

! Perceptron(algılayıcı): gerçek olan çıktı ile tahmin edilen çıktı arasındaki farkı alıp bu bilgiyi neural networke geri yansıtır. geri beslemek için kullanıyo => bu ne demek ağırlıkları değiştiriyo

! learning rate: perceptrondan gelen değerin(hata miktarının hesaplanmış hali), sisteme ne kadar yansıtılacağıdır. (ne kadar hızlı öğrenceğidir)

! Gradient Descendent(Gradyan alçalış): Yapay sinir ağının öğrenmesi aslında en temelede yapay sinir ağlarındaki (sinapsislerdeki) ağırlıkların tekrar güncellenmesiyle gerçekleşir ve amaç tabiiki en doğru ağırlıkları bulmaktır. Sistem budur. Çok büyük yapay sinir ağlarında weightlerin hesaplanması sinir ağına tekrar yansıtılması oldukça fazla hesap gerektiren bir durum. Bu hesaplarla yapay sinir ağının kararlı bir hale gelmesi (optimum=istenilen) oldukça fazla vakit alır. gradient descent için öğrenme oranı vardır - big learning rate - small learning rate. bu öğrenme oranları ile optimum noktayı (artık yapay sinir ağının öğrenmiş halini) bulmaya çalışırsın. Bu öğrenmiş hale gelene kadar yapay sinir ağı eğitimine devam ediyor. Bu eğitime devam etmesi sırasında bütün mesele nasıl atlama hali yapacağı. yapay sinir ağının doğru noktaya gitmesi için gradient descent ile yumuşak alçalma / hafif geçiş. ***Gradient descent bize bu öğrenme sürecinde yapmış olduğumuz hataların bize en derinde, en optimumda, en iyi noktaya götürmesi ile ilgili nasıl adım atacağımız, adım değerlerinin nasıl değişeceği ile ilgili yöntemin ismi.***

! Stochastic Gradient Descendent: Her bir verinin sonunda learning rate değerini arttırma/değiştirme/azaltma gibi kararlar alıyorsak bu stochastik gradient descendent'dir. Yani tüm veriye bakmadan örnek veriye (1 satıra) bakarak yaptığımız iş.

! Batch Gradient Descendent: Tüm verinin okunarak bunun neticesinde bir karar verme süreci.

! Mini Batch Gradient Descendent: Stochastic ile Batch yaklaşımı arasında kalan yaklaşım. 

    Algoritma Adımları (Backward Propagation)
Adım 1: Bütün ağı rastgele sayılarla (sıfıra yakın ama sıfırdan farklı) ilkendir.
Adım 2: Veri kümesnden ilk satır (her öznitelik bir nöron olacak şekilde) giriş katmanından verilir.
Adım 3: İleri yönlü yayılım yapılarak, YSA istenen sonucu verene kadar güncellenir.
Adım 4: Gerçek ve çıktı arasındaki fark alınarak hata (error) hesaplanır.
Adım 5: Geri yayılım yapılarak, her sinaps üzerindeki ağırlık, hatadan sorumlu olduğu miktarda değiştirilir. Değiştirme miktarı ayrıca öğrenme oranına da bağlıdır.
Adım 6: Adım 1-5 arasındaki adımları istenen sonucu elde edene kadar güncelle (reinforcement learning) veya eldeki bütün verileri ilgili ağda çalıştırdıktran sonra bir seferde güncelleme yap (batch learning)
Adım 7: Bütün eğitim kümesi çalıştırıldıktan sonra bir çağ/tur (epoch) tamamlanmış olur. Aynı veri kümeleri kullanılarak çağ/tur tekrarları yapılır.
"""