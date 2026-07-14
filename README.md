# twitter-airline-sentiment

## 1. Project Overview
This project classifies the sentiment of airline-related tweets as positive, neutral, or negative. It covers the complete machine learning workflow, including data preprocessing, model training, evaluation, and deployment as a REST API using FastAPI, Docker, and AWS.

## 2. Project Pipeline 
Raw Tweets -> Text Preprocessing -> Feature Extraction -> Baseline Model Comparison -> PyTorch Baseline -> Fine-Tune BERT -> FastAPI -> Docker -> AWS EC2 Deployment

## 3. Tech Stack
Python $\cdot$ Pandas $\cdot$ Scikit-learn $\cdot$ PyTorch $\cdot$ Hugging Face Transformers $\cdot$ FastAPI $\cdot$ Docker $\cdot$ AWS EC2 $\cdot$ Amazon ECR

## 4. Dataset
This project uses the **Twitter US Airline Sentiment** dataset from Kaggle. The dataset contains customer tweets directed at major U.S. airlines, with each tweet manually labeled according to its overall sentiment.

- **Source:** Kaggle – Twitter US Airline Sentiment
- **Number of tweets:** 14,640
- **Number of classes:** 3
  - Negative
  - Neutral
  - Positive

The dataset also includes additional metadata such as the airline name, confidence score, and the reason for negative sentiment. In this project, only the tweet text and sentiment labels are used for model training.
