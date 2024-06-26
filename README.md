# NLP: Sentiment analysis using different ML/DL tools

## Summary

This repository aims to test different Machine Learning (ML) tools for Natural Language Processing (NLP) sentiment analysis.
This test focuses on identifying the sentiment - positive or negative - of tweets.

## The repo

You can run everything with the **sentiment_analysis.ipynb** without needing a GPU.
If you have a GPU and would like, you can also run **bart_example.ipynb** to obtain your own results with BART.

The algorithms included are: 

- BART
- Long Short Term Memory Neural Network (LSTM NN)
- Naive Bayes 
- K-Nearest Neighbors (KNN)

Before using these algorithms we also preprocessed the tweets at the **preprocess.py**, using the following steps:

- Removed urls
- Removed special characters and numbers
- Removed stop words in the english vocabulary
- Removed words shorter than 2 or 3 characters
- Stemmed all the words (Reduced inflected form of a word to a “stem,” or root form, also known as a “lemma” in linguistics.)

## The data

We use a version of the Twitter sentiment dataset ([link](https://drive.google.com/file/d/13mAaFqCrscUYkoITf4rZ6qG9ptAlIJVb/view?usp=sharing)).
We also compare the results with those obtained using Zero-shot prediction with a pre-trained [BART model](https://huggingface.co/transformers/model_doc/bart.html).

The repository contains a small version of the dataset ([data/twitter_dataset_small_w_bart_preds.csv](data/twitter_dataset_small_w_bart_preds.csv) containing 20K examples) with an additional column `bart_is_positive`, which contains the Zero-shot prediction of the BART model using the query `This example is positive`. 