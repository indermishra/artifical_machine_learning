import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report
data = {'Age': [20, 30, 35, 22, 40, 45, 50, 60, 52, 39], 'Monthly_recharge': [50, 70, 90, 110, 60, 120, 150, 170, 190, 340], 'churn': [1,0,1,1,0,0,1,0,1,0]}
df = pd.DataFrame(data)
print(df)
x = df[['Age', 'Monthly_recharge']]
y = df['churn']
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
svc_model = SVC(kernel='linear', C=1.0)
svc_model.fit(x_train, y_train)
user_age = float(input("Enter user Age: "))
user_monthly_recharge = float(input("Enter monthly Recharge: "))
predicted_churn = svc_model.predict([[user_age, user_monthly_recharge]])
print(predicted_churn[0])