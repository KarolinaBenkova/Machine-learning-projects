# Projects in Machine Learning

This repository contains three ML projects in Python, all done using the `scikit-learn` library.

## Modelling house prices

This project uses data on house sales from a large US city in order to develop a predictive model for the sale prices of homes in this city. Besides the sale price, the data include 15 features that provide additional details on the property. The data are provided in three files:

- `sales.csv`, used as the training data set,
- `sales_test.csv`, used as the testing data set, and
- `sales_holdout.csv` which is a copy of the testing data set and can be replaced by unseen data (used for assessment),

and the project can be found in the Python notebook `house_prices.ipynb`.

The goal is to create a model that predicts an accurate price for new houses coming to the market based on their properties. The method chosen for this project was **Ridge Regression** as it was found that it both performed well and would be easy to explain to both house buyers and sellers. First, the data is analysed and preprocessed, then the model is developed by tuning the penalization parameter using `GridSearchCV`. Finally, a summary of the model and a discussion about its results is provided.

## Assessing wine quality  

Here data from a paper published by Cortez, et al. in Decision Support Systems in 2009 were used to explore physicochemical characteristics of portuguese wine and their impact on wine's quality. Besides the quality, which is a score between 0 (worst) and 10 (best), 12 more characteristics are provided in the following datasets:

- `wine_qual_train.csv`- the training data set,
- `wine_qual_test.csv` - the testing data set,
- `wine_qual_holdout.csv` - copy of the testing data set, replaced by unseen data for the assessment.

The project can be found in the Python notebook `wine_quality.pdf`.

The aim of the project was to develop a classification model to predict quality of wine based on its characteristics. Rather than using scores for wine quality, the values were discretised into 4 categories (excellent, good, average, and poor). After considering a set of models, **Random Forest classifier** was chosen for this project. After initial data analysis and preprocessing, the parameters of the classifier are tuned using the classification report and the confusion matrix. In the discussion part, the model is summarised and its results are analysed.

## Categorising handwritten digist from the MNIST dataset

In this short project, both supervised and unsupervised learning techniques were explored on the MNIST dataset. The dataset was imported from the `tensorflow` library with 60,000 training images and 10,000 testing images of handwritten digits (from 0 to 9). The project can be found in `MNIST.ipynb`.

In the supervised learning part of this project, the goal is to predict an output, i.e. classify a handwritten digit, by learning the digits in the images with their correct labels, which is done by using the **k-Nearest Neighbours** method. Here we also tweak the parameters of the classifier and use the classification report with the confusion matrix.

Clustering as an unsupervised learning method consists in finding patterns among the data and grouping them into clusters. This was implemented using the **K-Means** method. Besides having 10 clusters, we also experiment with larger and smaller number of clusters.


- 
