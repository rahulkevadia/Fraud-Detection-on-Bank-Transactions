# Fraud-Detection-on-Bank-Transaction-Dataset

Software And Tools Requirements
1. [Github](https://github.com)
2. [VSCodeIDE](https://code.visualstudio.com)

Create a new environment

```
conda create -p venv python==3.7 -y
```

Git steps:
1. To check the status, whether green or red. Green means file/folder is staged, red means needs to staged.
```
git status
``` 
2. To add everything to staged
```
Git add .
```
3. To add a message, and push it to staging env from local.
```
git commit -m "commit-message" 
```
4. To pull the origin destination project.
```
git pull origin main
```
5. To upload on GitHub and push it to main branch.
```
git push origin main
```

Project Overview:

Detecting fraud in payments is a significant focus for cyber-crime agencies. Recent studies show machine learning can effectively identify fraudulent transactions in large payment datasets. These methods can catch fraud that humans might miss and do so in real time. 
In this project, I use several supervised machine-learning techniques to detect fraud in a publicly available dataset of simulated payment transactions. I aim to show how these techniques can accurately classify data even when fraudulent transactions are much rarer than normal ones. 
I have demonstrated how exploratory data analysis can help distinguish between fraudulent and non-fraudulent transactions. Additionally, I have shown that tree-based algorithms like Random Forest perform significantly better than Logistic Regression for datasets where fraud and non-fraud transactions are separable.

The dataset contains 11 columns for ~6 million rows of data. The key columns available are 
1. Type of transactions,
2. Amount transacted,
3. Customer ID and Recipient ID,
4. Old and New balance of Customer and Recipient,
5. Time step of the transaction,
6. Whether the transaction was fraudulent or not.

Project Plan:

*Data Cleaning 
   ** Type Conversion
   1.2. Summary Statistics
   1.3. Missing Values Check
2. Exploratory Analysis
   2.1. Class Imbalance
   2.2. Types of Transactions
   2.3. Data Sanity Check
       2.3.1. Negative or Zero Transaction Amount
       2.3.2. Originator Balance and Recipient Balance
       2.3.3. Fraud Transaction Analysis
3. Predictive Modeling for Fraud Detection
   3.1. Model Data Creation
   3.2. Standardization
   3.3. Data Splitting into Train and Test
4. Classification Models for Fraud Detection
   4.1. Logistic Regression
   4.2. Random Forest
5. Predicting on test Dataset.
6. Performance Metric.
7. Hyperparameter Tuning.
8. Pickling The Model file For Deployment

