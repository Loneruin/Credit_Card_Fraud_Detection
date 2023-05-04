# [Credit Fraud Detection](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)

## Summary

The dataset contains transaction made by credit cards in September 2013 by european cardholders. This dataset presents transactions information that occurred over a period of two days, with 492 cases of fraud out of 284,807 transactions. The dataset is heavily skewed towards non-fraudulent transactions, with the positive class (frauds) accounting for 0.172% of all transactions.

The input variables in the dataset are all numerical and have been obtained through a PCA transformation. However, due to confidentiality converns, the original features and additional contextual information are not provided.

The features V1 through V28 represent the principal components derived from the PCA transformation, while the features Time and Amount have not been transformed. The Time feature denotes the number of seconds elapsed between each transaction and the first transaction in the dataset, while the Amount feature represents the monetary value of each transaction. The response variable, Class, take a value of 1 in case of fraud and 0 otherwise. The amount feature can be used for cost-sensitive learning, depending on the specific machine learning algorithm chosen.

## Data Exploratory

[Credit Card Transaction Density](output/credit_card_transaction_time_density)
