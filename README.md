# Topic Modelling
Allows the user to :
1. Find out the best number of topics the document should be divided into using 
   1.1 [Weighted Jaccard Average Stability](https://rpubs.com/lgadar/weighted-jaccard) : Similarity index 
   1.2 [Symmetric Kullback Liebler divergence](https://bigdatascientistblog.wordpress.com/2017/09/11/a-simple-introduction-to-kullback-leibler-divergence-through-python-code/) : Dis-similarity index 
2. Prepare a list of the topics and their description
3. Plot top words
4. Evolution of the frequencies of the topic (topic vs time)


## Features

### Vector space modeling
-  Feature selection based on word frequency
-  Weighting
    - tf
    - tf-idf


### Load and prepare a textual corpus

A corpus is a TSV(i.e. tab separated values) file describing documents, formatted as follows: a document per line, with at least three columns, namely id (a number), title (a short text) and text (the full content of the document), e.g.:

```
id	title	text
1	Document 1's title	This is the full content of document 1.
2	Document 2's title	This is the full content of document 2.
etc.
```

The following code snippet shows how to load a corpus of documents and vectorize them using tf-idf with unigrams. One just needs to change the input file and then there will be textual corpus for further analysis. The language can be changed as well. 

```
corpus = Corpus(source_file_path='input/x.csv',
                language='english', 
                vectorization='tfidf', 
                n_gram=1,
                max_relative_frequency=0.8, 
                min_absolute_frequency=4)
print('corpus size:', corpus.size)
print('vocabulary size:', len(corpus.vocabulary))
print('Vector representation of document 0:\n', corpus.vector_for_document(0))
```

### Instantiate a topic model and infer topics

It is possible to instantiate a NMF or LDA object then infer topics. 

NMF:

```
topic_model = NonNegativeMatrixFactorization(corpus)
topic_model.infer_topics(num_topics=15)
```

LDA (using either the standard variational Bayesian inference or Gibbs sampling):

```
topic_model = LatentDirichletAllocation(corpus)
topic_model.infer_topics(num_topics=15, algorithm='variational')
```
```
topic_model = LatentDirichletAllocation(corpus)
topic_model.infer_topics(num_topics=15, algorithm='gibbs')
```



