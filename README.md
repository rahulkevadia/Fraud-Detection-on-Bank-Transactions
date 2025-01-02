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

The dataset contains 11 columns for ~6 million rows of data. The key columns available are Type of transactions, Amount transacted, Customer ID and Recipient ID, Old and New balance of Customer and Recipient, Time step of the transaction, Whether the transaction was fraudulent or not.
