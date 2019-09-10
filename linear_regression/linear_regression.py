import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


data = pd.read_csv('weight-height.csv')
#read in data
data.head()

XY = data.values
#convert data to array and slice
X = np.array(XY[:,2])
Y = np.array(XY[:,1])

print(X)
print(Y)

plt.scatter(X,Y)
#plot a scatter plot of the X and Y values
plt.show()

denominator = X.dot(X) - X.mean() * X.sum()
a = (X.dot(Y) - Y.mean() * X.sum()) / denominator
b = (Y.mean() * X.dot(X) - X.mean() * X.dot(Y)) / denominator

Y_hat = a*X + b
#formula for equation of a line where a is the slope, X is the input, b is the y-intercept and Yhat is the predicted 
#output.

plt.scatter(X,Y)
plt.plot(X, Y_hat, color='red')
#plot the line of best fit
plt.show()



# add bias term
data['ones'] = 1
X_new = data[['Weight', 'ones']]
print(X_new)
Y_new = data['Height']
print(Y_new)
w = np.linalg.solve(np.dot(X_new.T, X_new), np.dot(X_new.T, Y_new))
#using maximum likelihood to predict Y
Y_hat_2 = X_new.dot(w)


plt.scatter(X,Y)
plt.plot(X, Y_hat_2, color='green')
#plot the new line of best fit
plt.show()

print (w)
print (Y_hat_2)


ss_res = Y - Y_hat
ss_tot = Y - Y.mean()
r_squared = 1 - (ss_res.dot(ss_res)/ss_tot.dot(ss_tot))
#to check the performance of our model
print ("The R squared value is:", r_squared)

ss_res_2 = Y - Y_hat_2
r_squared_2 = 1 -((ss_res_2.dot(ss_res_2)/ss_tot.dot(ss_tot)))
#to check the performance of our model with bias and maximum likelihood
print ("r_squared for new Y_hat:", r_squared_2)


