Churn Data Analysis



Predicting customer churn for a telecom company using classification models, with a deployed interactive app for live predictions.



🔗 Live App: Try it here https://customerchurnapp-mnrflnnqdom3avjaeb4kre.streamlit.app/ |  📄 Full Report: REPORT.md



Summary



A tuned Random Forest model predicts which customers are likely to churn, trained on the Telco Customer Churn dataset (7,043 customers, 26.5% churn rate).



Metric	Value

ROC-AUC	0.8457

Recall (Churn)	0.78

Precision (Churn)	0.55



The model correctly identifies 73% of actual churners in the test set, with the top predictive signals being tenure, monthly/total charges, and having a fiber-optic + month-to-month contract combination — customers who are new, high-spending, and on flexible contracts are the highest risk group.



What's in this repo

File	Purpose

Churn.ipynb	Full training notebook — EDA, preprocessing, feature engineering, model tuning, evaluation

app.py	Streamlit app for interactive churn prediction

predict.py	Standalone prediction script (no notebook dependency)

model\_artifacts/	Saved model, scaler, and feature metadata needed for inference

requirements.txt	Python dependencies

REPORT.md	Full technical report — methodology, all model comparisons, evaluation detail

Approach

Cleaned \& encoded the raw data; engineered AvgMonthlySpend, TenureBucket, and a Fiber\_x\_MonthToMonth interaction feature.

Compared Logistic Regression, Random Forest, and XGBoost as baselines.

Tuned Random Forest and XGBoost via GridSearchCV, using class-weighting instead of SMOTE to handle the \~26.5% churn rate.

Selected the tuned Random Forest — best ROC-AUC and precision, with a recall gap to XGBoost (0.78 vs. 0.79) too small to be meaningful.

Deployed the model behind a Streamlit form for live predictions.

