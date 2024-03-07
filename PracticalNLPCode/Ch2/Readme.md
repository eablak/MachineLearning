![alt text](https://raw.githubusercontent.com/practical-nlp/practical-nlp-figures/master/figures/2-1.png)

**Pre-Processing**

__Preliminaries__

* Sentence segmentation and word tokenization

Any NLP pipeline has to start with a reliable system to split the text into sentences (sentence segmentation) and further split a sentence into words (word tokenization).

![alt text](https://raw.githubusercontent.com/practical-nlp/practical-nlp-figures/master/figures/2-6.png)


__Frequent steps__

* Stop word removal, stemming and lemmatization, removing digits/punctuation, lowercasing, etc.

Assume we have a good sentence segmenter and word tokenizer in place. At that point, we would have to start thinking about what kind of information is useful for developing a categorization tool. Some of the frequently used words in English, such as a, an, the, of, in, etc., are not particularly useful for this task, as they don’t carry any content on their own to separate between the four categories. Such words are called stop words and are typically (though not always) removed from further analysis in such problem scenarios. There is no standard list of stop words for English, though.

Stemming refers to the process of removing suffixes and reducing a word to some base form such that all different variants of that word can be represented by the same form (e.g., “car” and “cars” are both reduced to “car”).

Lemmatization is the process of mapping all the different forms of a word to its base word, or lemma. While this seems close to the definition of stemming, they are, in fact, different. For example, the adjective “better,” when stemmed, remains the same. However, upon lemmatization, this should become “good,”.

![alt text](https://raw.githubusercontent.com/practical-nlp/practical-nlp-figures/master/figures/2-7.png)


![alt text](https://raw.githubusercontent.com/practical-nlp/practical-nlp-figures/master/figures/2-8.png)


__Other steps__

* Normalization, language detection, code mixing, transliteration, etc.

When we’re working on developing NLP tools to work with such data, it’s useful to reach a canonical representation of text that captures all these variations into one representation. This is known as text normalization. Some common steps for text normalization are to convert all text to lowercase or uppercase, convert digits to text (e.g., 9 to nine), expand abbreviations, and so on.

![alt text](https://raw.githubusercontent.com/practical-nlp/practical-nlp-figures/master/figures/2-9.png)

Code mixing refers to this phenomenon of switching between languages.


__Advanced processing__

POS tagging, parsing, coreference resolution, etc.

![alt text](https://raw.githubusercontent.com/practical-nlp/practical-nlp-figures/master/figures/2-10.png)

![alt text](https://raw.githubusercontent.com/practical-nlp/practical-nlp-figures/master/figures/2-11.png)


**Feature Engineering**

The goal of feature engineering is to capture the characteristics of the text into a numeric vector that can be understood by the ML algorithms. Two different approaches taken in practice for feature engineering in (1) a classical NLP and traditional ML pipeline and (2) a DL pipeline.

![alt text](https://raw.githubusercontent.com/practical-nlp/practical-nlp-figures/master/figures/2-12.png)

The main takeaway for building classical ML models is that the features are heavily inspired by the task at hand as well as domain knowledge (for example, using sentiment words in the review example). One of the advantages of handcrafted features is that the model remains interpretable—it’s possible to quantify exactly how much each feature is influencing the model prediction.

A noisy or unrelated feature can potentially harm the model’s performance by adding more randomness to the data. Recently, with the advent of DL models, this approach has changed. In the DL pipeline, the raw data (after preprocessing) is directly fed to a model. The model is capable of “learning” features from the data. Hence, these features are more in line with the task at hand, so they generally give improved performance. But, since all these features are learned via model parameters, the model loses interpretability.