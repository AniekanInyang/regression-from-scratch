import numpy as np
import pandas as pd
import matplotlib.pyplot as plt



data = pd.read_csv("weight-height.csv")
#read the data
data = data.sample(frac=1)
#select random data samples
XY = data.values
#converting the data to arrays
XY[:, 0] = [0 if x=="Male" else 1 for x in XY[:, 0]]
#converting the labels(Y) to 0 and 1
XY[:, 1] = (XY[:,1] - XY[:,1].mean())/XY[:,1].std()
XY[:, 2] = (XY[:,2] - XY[:,2].mean())/XY[:,2].std()
#Standardizing the X values (features)
XY = XY[:1000, :]
#reducing the dataset to 1000 rows
print(XY)


train = XY[:500,:]
test = XY[:-500, :]

X_train = train[:, 1:3]
Y_train = train[:, 0]

X_test = test[:, 1:]
Y_test = test[:, 0]

ones = np.array([[1] * 500]).T
#a 500 x 1 2D maxtrix of ones
X_train = np.concatenate((ones, X_train), axis=1)
#adding bias to the train data

print(X_train)

fig = plt.figure()
plt.scatter(X_train[:,0], Y_train)
plt.scatter(X_train[:, 1], Y_train)
plt.show()
#plotting the features against Y


#Number 3
def sigmoid(z):
        return 1/(1 + np.exp(-z))

theta = np.random.randn((3))
#initializing random weights

z = np.array(X_train.dot(theta), dtype=np.float32)
Y = sigmoid(z)


def loss(h, Y_train):
#loss function
        return (Y_train * np.log(h) - (1 - Y_train) * np.log(1 - h)).mean()

lr = 0.001
#learning rate of 0.001
for i in range(500000):
#using sigmoid function to obtain the loss on the train data in 500000 steps and a learning rate of 0.001
        z = np.array(np.dot(X_train, theta), dtype=np.float32)
        h = sigmoid(z)
        gradient = np.dot(X_train.T, (h - Y_train)) / Y_train.shape[0]
        theta = theta - (lr * gradient)

        if i % 10000 == 0:
                print (f'loss: {loss(h, Y_train)} \t')


print ('final weight:', theta)    
print ('final loss:', loss(h, Y_train))


test_preds = []
weight = np.random.randn((2))
z = np.array(np.dot(X_test, weight), dtype=np.float32)
for i in sigmoid(z):
        if i > 0.5:
                test_preds.append(1)
        if i < 0.5:
                test_preds.append(0)

print (test_preds)
print(Y_test)

acc = (test_preds == Y_test).mean()
#checking the accuracy of the model on test_data
print(acc)