# BITS Pilani M.Tech (AIML) - Machine Learning Assignment 2

## a. Problem Statement
To build, evaluate, and deploy multiple classification models using an end-to-end Machine Learning workflow, culminating in an interactive Streamlit web application.

## b. Dataset Description
* **Dataset Used:** Portuguese Student Achievement Dataset (`student-por.csv`) sourced from the UCI Machine Learning Repository.
* **Instances:** 649 rows (Satisfies the requirement of $\ge 500$ instances)[cite: 2].
* **Features:** 30+ attributes (Satisfies the requirement of $\ge 12$ features)[cite: 2].
* **Target Variable:** Binary classification task predicting student pass/fail status based on final grade (`G3`).

## c. GitHub Repository Link
https://github.com/nehaarora-arch/bits-ml-assignment-2

## d. Models Used & Evaluation Comparison Table

           ML Model Name  Accuracy     AUC  Precision  Recall      F1     MCC
0     Logistic Regression    0.8769  0.7843     0.9231  0.9391  0.9310  0.3612
1           Decision Tree    0.8231  0.6101     0.9107  0.8870  0.8987  0.2038
2                     KNN    0.8692  0.7670     0.8828  0.9826  0.9300 -0.0451
3             Naive Bayes    0.8154  0.7217     0.9417  0.8435  0.8899  0.3493
4  Support Vector Machine    0.8923  0.8487     0.9040  0.9826  0.9417  0.3034


### Observations on Model Performance
| ML Model Name | Observation about model performance |
| :--- | :--- |
| **Logistic Regression** | Stable linear boundary, highly interpretable, achieving strong accuracy (0.8769) and balanced precision/recall.|
| **Decision Tree** | Lower AUC (0.6101) and accuracy (0.8231), indicating minor overfitting and difficulty generalizing complex splits. |
| **KNN** | High recall (0.9826), but a negative MCC score (-0.0451) highlights sensitivity to distance metrics and class distribution. |
| **Naive Bayes** | Fast computation with high precision (0.9417), though limited by strict feature independence assumptions. |
| **Support Vector Machine** | Achieves robust boundary separation with the highest overall accuracy (0.8923) and AUC (0.8487). |
| **Overall Winner** | **Support Vector Machine** (Secured the highest overall Accuracy of 89.23% and superior AUC performance). |
