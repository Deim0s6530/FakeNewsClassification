# FakeNewsClassification
This project focuses on the classification of fake news using advanced Natural Language Processing (NLP) techniques and machine learning and deep learning models. It covers the entire classification pipeline, from data exploration to deploying Transformer-based models.

Project Content
1. Data Exploration and Preprocessing
Loading and cleaning data from the WELFake_Dataset.csv dataset.
Handling missing values and removing unnecessary columns.
Merging the title and text columns for unified analysis.
Adding new metrics, such as the lengths of texts, titles, and article bodies.
2. Exploratory Data Analysis (EDA)
Analyzing distributions of fake and real news data.
Visualizing relationships between text/title lengths and labels.
Analyzing correlations between numerical variables.
3. Modeling
Text Vectorization:
Implementing TF-IDF to transform textual data into numerical vectors.
Using advanced techniques such as stemming to enhance data quality.
Training Classical Models:
Naive Bayes for establishing a baseline.
Advanced Transformer-Based Models:
Utilizing BERT for text classification.
Preparing data for BERT with steps like tokenization and sequence length management.
Implementing a custom Transformer model.
4. Training and Validation
Splitting data into training and testing sets.
Training models with PyTorch and validating them on separate datasets.
Using optimizers like AdamW and learning rate schedulers.
Technologies Used
Python
Pandas, NumPy, Matplotlib, Seaborn for data analysis and visualization.
Scikit-learn for classical models and TF-IDF vectorization.
PyTorch, Hugging Face Transformers for advanced models and Transformer handling.
BERT for text classification.
Objectives
Automatically identify fake news with high accuracy.
Compare performance between classical and Transformer-based models.
Provide a complete and reproducible pipeline for fake news classification