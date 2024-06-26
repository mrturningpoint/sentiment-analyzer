Reviews Sentiment Analysis

This project focuses on building a sentiment analysis model to classify restaurant reviews as positive or negative using Natural Language Processing (NLP) and machine learning techniques.

Overview
The project involves the following key steps:

Data Import and Cleaning:

Imported the dataset containing restaurant reviews from a TSV (Tab Separated Values) file.

Cleaned the reviews by removing non-alphabetic characters, converting text to lowercase, removing stopwords (except "not"), and applying stemming.

Text Preprocessing:


Created a corpus of cleaned reviews.

Transformed the text data into numerical features using the CountVectorizer from scikit-learn.
Dataset Splitting:

Split the dataset into training and test sets to evaluate the performance of the machine learning model.
Model Training and Prediction:


Trained a Kernel SVM (Support Vector Machine) model on the training set.

Predicted the sentiment of reviews in the test set.
Evaluation:

Evaluated the model using a confusion matrix and accuracy score.

Applied k-Fold Cross Validation to ensure the model's robustness and to compute the mean accuracy and standard deviation.

Installation
To run this project, ensure you have Python installed along with the following libraries:

numpy
matplotlib
pandas
nltk
scikit-learn