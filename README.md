# TOM


## Features

### Vector space modeling
-  Feature selection based on word frequency
-  Weighting
    - tf
    - tf-idf

## Usage

We provide two sample programs, topic_model.py (which shows you how to load and prepare a corpus, estimate the optimal number of topics, infer the topic model and then manipulate it) and topic_model_browser.py (which shows you how to generate a topic model browser to explore a corpus), to help you get started using TOM.

### Load and prepare a textual corpus

A corpus is a TSV (tab separated values) file describing documents, formatted as follows: a document per line, with at least three columns, namely id (a number), title (a short text) and text (the full content of the document), e.g.:

```
id	title	text
1	Document 1's title	This is the full content of document 1.
2	Document 2's title	This is the full content of document 2.
etc.
```

The following code snippet shows how to load a corpus of French documents and vectorize them using tf-idf with unigrams.

```
corpus = Corpus(source_file_path='input/raw_corpus.csv',
                language='french', 
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

### Instantiate a topic model and estimate the optimal number of topics

Here we instantiate a NMF object, then generate plots with the three metrics for estimating the optimal number of topics.

```
topic_model = NonNegativeMatrixFactorization(corpus)
viz = Visualization(topic_model)
viz.plot_greene_metric(min_num_topics=5, 
                       max_num_topics=50, 
                       tao=10, step=1, 
                       top_n_words=10)
viz.plot_arun_metric(min_num_topics=5, 
                     max_num_topics=50, 
                     iterations=10)
viz.plot_brunet_metric(min_num_topics=5, 
                       max_num_topics=50,
                       iterations=10)
```

### Save/load a topic model

To allow reusing previously learned topics models, TOM can save them on disk, as shown below.

```
utils.save_topic_model(topic_model, 'output/NMF_15topics.tom')
topic_model = utils.load_topic_model('output/NMF_15topics.tom')
```

### Print information about a topic model

This code excerpt illustrates how one can manipulate a topic model, e.g. get the topic distribution for a document or the word distribution for a topic.

```
print('\nTopics:')
topic_model.print_topics(num_words=10)
print('\nTopic distribution for document 0:',
      topic_model.topic_distribution_for_document(0))
print('\nMost likely topic for document 0:',
      topic_model.most_likely_topic_for_document(0))
print('\nFrequency of topics:',
      topic_model.topics_frequency())
print('\nTop 10 most relevant words for topic 2:',
      topic_model.top_words(2, 10))
```

