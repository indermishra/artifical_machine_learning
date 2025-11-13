''' Problem statement: 
"Predict student scores based on the number of hours they studied."
'''

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
#step 2 : Load the dataset
data = {'Hours_studyed': [1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0, 5.5, 6.0], 'Exam_Scores': [35, 40, 45, 50, 55, 60, 65, 70, 75, 80]}
df = pd.DataFrame(data)
print("Dataset:")
print(df)  
x = df[['Hours_studyed']]
y = df[['Exam_Scores']]

x_train, x_test, y_train, y_test = train_test_split(x,y, test_size=0.2, random_state=42)
model = LinearRegression()
model.fit(x_train, y_train)
user_input = float(input('Enter the number of hours you studyed:  '))
predected_std_score = model.predict([[user_input]])
print(f"Predicted student score: {predected_std_score[0][0]}")
