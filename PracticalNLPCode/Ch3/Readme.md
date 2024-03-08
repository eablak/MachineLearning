How do we go about doing feature engineering for text data? In other words, how do we transform a given text into numerical form so that it can be fed into NLP and ML algorithms?

![alt text](https://github.com/practical-nlp/practical-nlp-figures/raw/master/figures/3-1.png)

It’s clear that mathematically representing images, video, and speech is straightforward. What about text? It turns out that representing text is not straightforward.

Go all the way to state-of-the-art techniques for representing text. These approaches are classified into four categories:

1. Basic vectorization approaches

- One-Hot Encoding
- Bag of Words

The key idea behind it is as follows: represent the text under consideration as a bag (collection) of words while ignoring the order and context. The basic intuition behind it is that it assumes that the text belonging to a given class in the dataset is characterized by a unique set of words. If two text pieces have nearly the same words, then they belong to the same bag (class). Thus, by analyzing the words present in a piece of text, one can identify the class (bag) it belongs to.

- Bag of N-Grams

It does so by breaking text into chunks of n contiguous words (or tokens). This can help us capture some context, which earlier approaches could not do. Each chunk is called an n-gram. The corpus vocabulary, V, is then nothing but a collection of all unique n grams across the text corpus. Then, each document in the corpus is represented by a vector of length |V|. This vector simply contains the frequency counts of n-grams present in the document and zero for the n-grams that are not present.

- TF-IDF (term frequency–inverse document frequency)

It aims to quantify the importance of a given word relative to other words in the document and in the corpus. It’s a commonly used representation scheme for information-retrieval systems, for extracting relevant documents from a corpus for a given text query.


2. Distributed representations

- Word Embeddings

- Going Beyond Words

3. Universal language representation
4. Handcrafted features