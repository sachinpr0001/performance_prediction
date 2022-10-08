import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# ### Load and Visualize the Data
# - Download
# - Load
# - Visualize
# - Normalisation
# Load
X = pd.read_csv('./Training Data/Linear_X_Train.csv')
Y = pd.read_csv('./Training Data/Linear_Y_Train.csv')

# Convert X and Y to Numpy arrays
X = X.values
Y = Y.values

# Normalization
u = X.mean()
std = X.std()
print(u,std)
# the result show that std is almost equal to 1 but not 1. this means,
# std is almost normalized. let's normalize it completely.
X = (X-u)/std
print(u,std)

# Visualize
# plt.style.use('seaborn')
# plt.scatter(X,Y, color = 'orange')
# plt.title("Hardwork Vs Performance Graph")
# plt.xlabel("Hardwork")
# plt.ylabel('Performance')
# #plt.show()
# X.shape, Y.shape


# ### Section 2. Linear Regression

def hypothesis(x,theta):
    # theta = [theta0, theta1]
    y_ = theta[0] + theta[1]*x
    return y_

def gradient(X,Y, theta):
    m = X.shape[0]
    grad = np.zeros((2,))
    for i in range(m):
        x = X[i]
        y_ = hypothesis(x, theta)
        y = Y[i]
        grad[0] += (y_ - y)
        grad[1] += (y_ - y)*x
    return grad/m
    
def error(X,Y,theta):
    m = X.shape[0]
    total_error = 0.0
    for i in range(m):
        y_ = hypothesis(X[i], theta)
        total_error += (y_ - Y[i])**2
        
    return total_error/m

    
def gradientDescent(X,Y, max_steps = 100, learning_rate = 0.1):
    
    theta = np.zeros((2,))
    error_list = []
    theta_list = []
    
    
    for i in range(max_steps):
        
        # Compute Grad
        grad = gradient(X,Y,theta)
        e = error(X,Y,theta)
        # update theta
        theta[0] = theta[0] - learning_rate*grad[0]
        theta[1] = theta[1] - learning_rate*grad[1]
        # Starting the theta values during updates
        theta_list.append((theta[0], theta[1]))
        error_list.append(e)
    
    return theta, error_list, theta_list

theta, error_list, theta_list = gradientDescent(X,Y)

theta
error_list
#theta_list
plt.plot(error_list)
plt.title("Reduction in Error Over Time")
#plt.show()


# ### Section 3. Predictions and Best Line

y_ = hypothesis(X,theta)
print(y_)

# Training  + Predictions
plt.scatter(X,Y)
plt.plot(X,y_, color = 'orange', label = 'Prediction')
plt.legend()
#plt.show()

# Load the test data
X_test = pd.read_csv('./Test Cases/Linear_X_Test.csv').values
Y_test = hypothesis(X_test, theta)

print(Y_test)

df = pd.DataFrame(data=Y_test, columns=["y"])

df.to_csv('y_prediction.csv', index = False)

# ### Section 4. Computing Score

def r2_score(Y,Y_):
    # Instead of using a loop, np.sum is recommended as it is fast
    num = np.sum((Y-Y_)**2)
    denom = np.sum((Y-Y.mean())**2)
    score = (1-num/denom)
    return score*100

r2_score(Y,y_)


# #### Section 5 Visualising Loss Function, Gradient Descent, Theta Updates

# Loss Actually 
T0 = np.arange(-40,40,1)
T1 = np.arange(40,120,1)
T0,T1 = np.meshgrid(T0,T1)
J = np.zeros(T0.shape)
for i in range(J.shape[0]):
    for j in range(J.shape[1]):
        y_ = T1[i,j]*X + T0[i,j]
        J[i,j] = np.sum((Y-y_)**2)/Y.shape[0]
        
print(J.shape)
# we have the loss function for 80*80 points.

#Visualize the J (loss)
# fig = plt.figure()
axes1 = plt.axes(projection ='3d') 
axes1.plot_surface(T0,T1,J,cmap = 'rainbow')
#plt.show()


# Contour Plot
axes2 = plt.axes(projection ='3d') 
axes2.contour(T0,T1,J,cmap = 'rainbow')
#plt.show()


# # plot the changes in values of theta

theta_list = np.array(theta_list)
#theta_list
ys = []
for i in range(100):
    ys.append(i)
print(len(ys))
plt.plot(theta_list[:,0], ys,label='Theta0')
plt.plot(theta_list[:,1], ys, label='Theta1')
plt.legend()
#plt.show()


# ## Tranjectory Traced by Theda Updates in the Loss Function

# fig = plt.figure()
axes = plt.axes(projection='3d')
axes.plot_surface(T0,T1,J,cmap = 'rainbow')
axes.scatter(theta_list[:,0], theta_list[:,1], error_list)
#plt.show()

# fig = plt.figure()
axes = plt.axes(projection='3d')
axes.contour(T0,T1,J,cmap = 'rainbow')
axes.scatter(theta_list[:,0], theta_list[:,1], error_list)
#plt.show()

# 2D contour plot / Top View

plt.contour(T0,T1,J,cmap='rainbow')
plt.scatter(theta_list[:,0], theta_list[:,1])
#plt.show()
