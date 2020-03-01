# Project: Predict Missing Labels

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1zpEKQFZHznlCxCTD2gnZe8Xz0aaHsURW)

**Objective:** Given an article along with its metadata, predict binary classes.

**Dataset:** Contains 1959 rows of -> dict("title", "content", "meta_des", "sentiment", "label"), where labels are binary.
Since some of the labels are missing, I split the dataset into two-part, i.e., full data rows (training - 1225 samples) and missing data rows (to_predict - 734).

**Model:** Support Vector Machine (SVM) - is a supervised machine learning model that uses classification algorithms for two-group classification problems. SVM is a fast and dependable algorithm that performs very well with a limited amount of data, which makes it great for this case.
I have trained the SVM on 0.9 of the training data and used the rest to evaluate the training performance. After training, I have used the trained model to predict the data with missing labels. Since it only takes a minute to run the pipeline, one could train a couple of SVM on the different data features, e.g. title, meta_des and articles, get its corresponding predictions, and compile combined predictions from the three based on the most common outputs. The model accuracy scores is above 90%.

