from sklearn.linear_model import Ridge, Lasso
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
import numpy as np
import matplotlib.pyplot as plt

#Sample Data
X=np.array([[1400],[1600],[1700],[1875],[1100],[1550],[2350],[2450]])
Y=np.array([245000,312000,279000,308000,199000,219000,405000,324000])

X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.2,random_state=42)

#Ridge Regression
ridge_model=Ridge(alpha=1.0)
ridge_model.fit(X_train,Y_train)
ridge_pred=ridge_model.predict(X_test)
ridge_mse=mean_squared_error(Y_test,ridge_pred)
print("Ridge Mean: ",ridge_mse)
print("Ridge R² Score:", r2_score(Y_test, ridge_pred))


#Lasso
Lasso_model=Lasso(alpha=0.1)
Lasso_model.fit(X_train,Y_train)
lasso_pred=Lasso_model.predict(X_test)
Lasso_mse=mean_squared_error(Y_test,lasso_pred)
print("Lasso mean: ",Lasso_mse)
print("Lasso R² Score:", r2_score(Y_test, lasso_pred))

plt.scatter(X_test, Y_test, color='black', label='Actual')
plt.plot(X_test, ridge_pred, color='blue', label='Ridge')
plt.plot(X_test, lasso_pred, color='green', label='Lasso')
plt.legend()
plt.title("Ridge vs Lasso Predictions")
plt.show()