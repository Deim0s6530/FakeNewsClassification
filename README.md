# Fake News Detection

## Project Overview
This project focuses on detecting fake news using a variety of Machine Learning and Deep Learning algorithms. The notebooks include implementations of traditional algorithms (e.g., Naive Bayes, Decision Tree, Random Forest) and advanced models such as RNNs, LSTMs, CNNs, and Transformers (e.g., BERT).

## Notebooks Included

### `main.ipynb`
- **TFIDF**: Feature extraction using Term Frequency-Inverse Document Frequency.
- **Machine Learning Algorithms**: Implementation of:
  - Naive Bayes (NB)
  - Decision Tree
  - Random Forest
- **Deep Learning Algorithms**: Exploration of:
  - Recurrent Neural Networks (RNNs)
  - Long Short-Term Memory networks (LSTMs)
  - Convolutional Neural Networks (CNNs)
  - Bidirectional Encoder Representations from Transformers (BERT)

### `Transformers.ipynb`
- **TFIDF**: Revisiting feature extraction.
- **Preparing the Data**: Preprocessing steps to clean and organize the dataset.
- **Transformers**: Utilizing modern Transformer-based models for fake news detection.

## Dataset
The project uses the [Fake News Classification Dataset](https://www.kaggle.com/datasets/saurabhshahane/fake-news-classification/data) available on Kaggle. It analyzes the linguistic patterns in fake and real news, noting, for instance, that fake news articles generally have a higher word count compared to real news.

## Prerequisites
- Python 3.x
- Jupyter Notebook
- Required libraries (install via `requirements.txt`):
  - Scikit-learn
  - TensorFlow
  - PyTorch
  - Hugging Face Transformers
