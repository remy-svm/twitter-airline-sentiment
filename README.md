# twitter-airline-sentiment

## 1. Project Overview
This project classifies the sentiment of airline-related tweets as positive, neutral, or negative. It covers the complete machine learning workflow, including data preprocessing, model training, evaluation, and deployment as a REST API using FastAPI, Docker, and AWS.

## 2. Project Pipeline 
Raw Tweets -> Text Preprocessing -> Feature Extraction -> Baseline Model Comparison -> PyTorch Baseline -> Fine-Tune BERT -> FastAPI -> Docker -> AWS EC2 Deployment

## 3. Dataset
This project uses the **Twitter US Airline Sentiment** dataset from Kaggle. The dataset contains customer tweets directed at major U.S. airlines, with each tweet manually labeled according to its overall sentiment.

- **Source:** Kaggle – Twitter US Airline Sentiment
- **Number of tweets:** 14,640
- **Number of classes:** 3
  - Negative
  - Neutral
  - Positive

The dataset also includes additional metadata such as the airline name, confidence score, and the reason for negative sentiment. In this project, only the tweet text and sentiment labels are used for model training.

## 4. Exploratory Data Analysis
Before model development, exploratory data analysis (EDA) was performed to better understand the dataset.

The analysis included:

- Distribution of sentiment labels
- Distribution of tweets across airlines
- Most frequent words
- Word clouds for positive, neutral, and negative tweets
- Tweet length analysis

The EDA revealed that the dataset is imbalanced, with negative tweets representing the majority class. To ensure that performance on all three sentiment classes was considered equally, models were evaluated using macro precision, macro recall, and macro F1 score, in addition to accuracy. 

Representative visualizations from the exploratory data analysis are shown below.

### Sentiment Distribution
<img width="582" height="497" alt="Screenshot 2026-07-14 at 12 10 17 AM" src="https://github.com/user-attachments/assets/9915b9ba-1c22-4c8f-9543-8d97b86b61ac" />

### Word Clouds
<img width="507" height="867" alt="Screenshot 2026-07-14 at 12 12 50 AM" src="https://github.com/user-attachments/assets/db1d998b-f0cf-47a4-95f3-378da7d6de3b" />

## 4. Text Preprocessing
The tweet text was cleaned before feature extraction and model training to reduce noise and create a consistent representation of the data. The preprocessing pipeline included the following steps:

- Removed tweets with missing text.
- Converted all text to lowercase.
- Removed URLs and user mentions (`@username`).
- Removed the `#` symbol while preserving the associated hashtag word.
- Removed punctuation and numerical characters.
- Removed extra whitespace and trimmed leading/trailing spaces.
- Applied WordNet lemmatization to convert words to their base verb forms (e.g., *cancelled* → *cancel*).

The original tweet text was preserved separately for error analysis, while the processed text was used for feature extraction and model training.

## 5. Feature Extraction
Two text vectorization techniques were evaluated using **Logistic Regression** with **5-fold cross-validation** to determine the most effective feature representation for the classical machine learning models.

| Feature Extraction | Accuracy | Macro Precision | Macro Recall | Macro F1 |
|-------------------|---------:|----------------:|-------------:|---------:|
| CountVectorizer | 0.7875 | 0.7385 | **0.7190** | **0.7271** |
| TF-IDF | **0.7915** | **0.7741** | 0.6797 | 0.7133 |

Although TF-IDF achieved slightly higher accuracy and macro precision, **CountVectorizer** produced the highest **Macro F1 score**, which was selected as the primary evaluation metric for comparing feature representations. Therefore, CountVectorizer was used for the subsequent classical machine learning experiments.

## 6. Baseline Classical Machine Learning Models

### Model Selection

After selecting **CountVectorizer** as the feature extraction method, five classical machine learning models were evaluated using **5-fold cross-validation**:

- Logistic Regression
- Naive Bayes
- Linear Support Vector Machine (Linear SVM)
- Random Forest
- XGBoost

Model performance was evaluated using **Accuracy**, **Macro Precision**, **Macro Recall**, and **Macro F1 Score**.

| Model | Accuracy | Macro Precision | Macro Recall | Macro F1 |
|-------|---------:|----------------:|-------------:|---------:|
| Logistic Regression | **0.7929** | 0.7434 | **0.7274** | **0.7349** |
| Naive Bayes | 0.7687 | 0.7075 | 0.7162 | 0.7112 |
| Linear SVM | 0.7623 | 0.7010 | 0.7005 | 0.7005 |
| Random Forest | 0.7687 | 0.7485 | 0.6360 | 0.6709 |
| XGBoost | 0.7920 | **0.7571** | 0.7015 | 0.7244 |

**Logistic Regression** achieved the highest **Macro F1 score** while also obtaining the highest accuracy and macro recall among the evaluated models. Consequently, Logistic Regression was selected as the strongest classical machine learning baseline for comparison with the subsequent deep learning models.

### Error Analysis

To better understand the strengths and limitations of the baseline model, a confusion matrix, classification report, and misclassified tweets were analyzed.

| Actual / Predicted | Negative | Neutral | Positive |
|--------------------|---------:|--------:|---------:|
| **Negative** | 1522 | 222 | 91 |
| **Neutral** | 209 | 355 | 56 |
| **Positive** | 115 | 84 | 274 |

The confusion matrix shows that **negative tweets were classified most accurately**, achieving an F1 score of **0.83**. Performance was lower for **neutral** (F1 = **0.55**) and **positive** (F1 = **0.61**) tweets, indicating that these classes are more difficult to distinguish.

Inspection of misclassified tweets revealed several common sources of error:

- **Mixed sentiment:** Tweets containing both positive and negative opinions (e.g., flight delays accompanied by praise for the crew).
- **Short or ambiguous messages:** Very short tweets often lacked enough context for reliable classification.
- **Context-dependent language:** Questions, travel updates, and customer service conversations were frequently predicted as neutral despite expressing sentiment.
- **Implicit sentiment:** Some tweets expressed frustration or satisfaction without using strong sentiment words, making them difficult for a bag-of-words model to classify correctly.

These observations highlight a key limitation of classical bag-of-words models: they represent words independently and cannot capture contextual relationships or nuanced language. This motivated the use of a deep learning model, followed by fine-tuning a pretrained BERT model to better model contextual information.

## 7. PyTorch Baseline Model

To establish a deep learning baseline, a simple neural network was implemented in **PyTorch**. The model consists of an **embedding layer**, **mean pooling**, and a **fully connected classification layer**. To improve generalization, **early stopping** was introduced using the **validation Macro F1 score** as the monitoring metric.

- **Patience:** 8 epochs
- **Best validation metric:** Macro F1
- **Training stopped at:** Epoch 28

The best-performing model weights were restored before evaluation on the test set.

### Final Performance

| Metric | Score |
|--------|------:|
| Accuracy | **0.77** |
| Macro Precision | **0.70** |
| Macro Recall | **0.72** |
| Macro F1 | **0.71** |


