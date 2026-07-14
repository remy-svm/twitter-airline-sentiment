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

### Sentiment Distribution
<img width="582" height="497" alt="Screenshot 2026-07-14 at 12 10 17 AM" src="https://github.com/user-attachments/assets/9915b9ba-1c22-4c8f-9543-8d97b86b61ac" />

### Word Clouds
<img width="507" height="867" alt="Screenshot 2026-07-14 at 12 12 50 AM" src="https://github.com/user-attachments/assets/db1d998b-f0cf-47a4-95f3-378da7d6de3b" />




## 4. Classic ML Results



