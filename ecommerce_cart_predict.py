import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

x = np.array([[20, 30, 1], [25, 40, 0], [30, 45, 1], [35, 50, 0]])
y = np.array([1,1,0,0])
x_train, x_test, y_train,_y_test = train_test_split(x, y, test_size=0.2, random_state=42)
model = LogisticRegression()
model.fit(x_train, y_train)
user_age = float(input("Enter user age: "))
user_stay_hours = float(input("Enter user hours to stay: "))
user_added_to_cart = int(input("Enter 1 if added to cart else Enter 0: "))
accuracy = model.score(x_train, y_train)
print(accuracy)
user_data = np.array([user_age, user_stay_hours, user_added_to_cart])
add_cart_predict = model.predict([user_data])
print(f"Add to cart prediction: {add_cart_predict[0]}")

if add_cart_predict[0] == 1:
    print("User will purchase the Product")
else:
    print("User will not purchase the Product")
    