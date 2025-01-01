# Fake News Classification with Transformers

This project focuses on the classification of fake news using advanced Transformer-based models. The implementation leverages **BERT** and custom Transformer architectures to classify textual data as fake or real.

## Dataset
The dataset used in this project is the **WELFake Dataset**. Due to size constraints, the dataset is not included in the repository. You can download it from Kaggle:

[Download WELFake Dataset from Kaggle](https://www.kaggle.com/datasets/marcodelarosa/welfake-dataset)

Please ensure the dataset is placed in the appropriate directory (`data/`) before running the scripts.

## Project Content

### 1. Data Exploration and Preprocessing
- Cleaning and preparing the dataset by removing unnecessary columns and handling missing values.
- Merging the `title` and `text` columns for unified analysis.
- Adding new metrics such as the lengths of texts, titles, and article bodies.

### 2. Preparing Data for Transformers
- Tokenizing textual data using **BERT Tokenizer**.
- Managing sequence lengths with padding and truncation.
- Creating attention masks for effective Transformer input processing.

### 3. Modeling
#### Pretrained Transformer Model (BERT):
- Fine-tuning **BERT** for binary classification (fake vs real news).
- Implementing BERT tokenization and encoding pipeline.
- Using PyTorch for training and validation.

#### Custom Transformer Model:
- Building a custom Transformer model for sequence classification.
- Defining hyperparameters such as `vocab_size`, `d_model`, `nhead`, and `num_encoder_layers`.

### 4. Training and Validation
- Splitting data into training, validation, and testing sets.
- Utilizing optimizers like **AdamW** and learning rate schedulers for effective training.
- Implementing loss functions like **CrossEntropyLoss**.
- Training on GPU for faster computation.

### 5. Evaluation
- Validating models on separate datasets.
- Comparing the performance of BERT and custom Transformer models.

## Technologies Used
- **Python**
- **Pandas**, **NumPy** for data preprocessing and manipulation.
- **Matplotlib**, **Seaborn** for data visualization.
- **PyTorch**, **Hugging Face Transformers** for model implementation and fine-tuning.
- **BERT** for pretrained Transformer-based text classification.

## Objectives
- Develop a high-performing fake news detection model using Transformer-based techniques.
- Compare the performance of pretrained BERT and custom Transformer architectures.
- Provide a reproducible pipeline for fake news classification.

---

### Instructions
1. Clone the repository.
2. Download the dataset from Kaggle and place it in the `data/` folder.
3. Install required dependencies using:
   ```bash
   pip install -r requirements.txt
