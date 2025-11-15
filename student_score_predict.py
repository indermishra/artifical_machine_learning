''' Problem statement: 
"Predict student scores based on the number of hours they studied."
'''

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
#step 2 : Load the dataset
# data = {'Hours_studyed': [1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0, 5.5, 6.0], 'Exam_Scores': [35, 40, 45, 50, 55, 60, 65, 70, 75, 80]}
# df = pd.DataFrame(data)
df = pd.read_csv('student_exam_data_new.csv')
print("Dataset:")
print(df)  
x = np.array([df['Study Hours'], df['Previous Exam Score']])
print("Features:  ")
print(x)
y = df['Pass/Fail']
print("Labels:  ")
print(y)
x = x.transpose()   

x_train, x_test, y_train, y_test = train_test_split(x,y, test_size=0.2, random_state=42)
model = LinearRegression()
model.fit(x_train, y_train)
user_input = float(input('Enter the number of hours you studyed:  '))
user_input_score = float(input('Enter the number of score :  '))
predected_status = model.predict([[user_input, user_input_score]])  # Assuming a previous exam score of 34
print(f"Predicted student status: {predected_status[0]:.2f} (1: Pass, 0: Fail)")
if predected_status[0] >= 1:
    print("The student is likely to Pass.")
else:
    print("The student is likely to Fail.")