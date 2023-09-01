# -*- coding: utf-8 -*-
"""
Created on Mon Jul  3 23:03:50 2023

@author: rd
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.pyplot import cm as color_cm
from matplotlib.cbook import flatten
import seaborn as sbn
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression



"""
ML: Normal Equation
One simple example for Linear Regression
theta = Xtrans.X.Xtrans.y
In this method nicely works data with small features
If it 1E5 or 6 it is a problem to find inv of matrix 
"""

_x = 2 * np.random.rand(100, 1)
_y = 4 + 3 * _x + np.random.rand(100, 1)
#In order to not to loose dimentionality, adding the column or np.c_[]  
x_b = np.c_[np.ones((100, 1)), _x]
theta_best = np.linalg.inv(x_b.T.dot(x_b)).dot(x_b.T).dot(_y)

#using theta fit to original data

x_new = np.array([[0], [2]])
x_new_b = np.c_[np.ones((2, 1)), x_new]
y_predict = x_new_b.dot(theta_best)

plt.plot(_x, _y, "*")
plt.plot(x_new, y_predict, "r-")
plt.xlim([0,2])
plt.ylim([0, 15])
plt.title("L-NORMALFIT")
plt.xlabel("X")
plt.ylabel("Y")
plt.show()

"""
It is done here. Next show will be using GD method for fitting data
the Batch GD, probably is wrong name. It should full GD. If you have large amont of data,
it will take longer time. In contrast Stochestic GD and Mini-batch GD is more effiect than 
Batch GD.
EXAMPLE
"""
#GD grad(MSE) = 2/sum(theta**T*X - y)*x theta and X are matric tensor, in which 
#each row is represented feature. step = theta - eta*grad(MSE) /eta: learning rate
# etas = np.linspace(1E-4, 1, 5);
etas = [0.02, 0.1, 0.5]
n_iter = 100; m = 100; theta_best = np.random.rand(2, 1)
theta_best_eta = []; x_new_b = []; y_predict = []
x_new = np.array([[0], [2]])
for i in range(len(etas)):
    for itera in range(n_iter):
        gradients = 2/m * x_b.T.dot(x_b.dot(theta_best) - _y)
        theta_best = theta_best - etas[i] * gradients
    theta_best_eta.append(theta_best)
    x_new_b.append(np.c_[np.ones((2, 1)), x_new])
    y_predict.append( x_new_b[i].dot(theta_best))
    

#Learning pandas data frame is more usefull than numpy.     

df = pd.DataFrame({"X":list(flatten(np.tile(x_new, (3, 1)))), "Y":list(flatten(np.array(y_predict)[:,:,0]))})
list_pd = [ g for i, g in df.groupby(df.index // 2)]
color = "rgb"
fig0, ax = plt.subplots(1, 4, figsize=(21,14))  
for i in range(len(list_pd)):
    # ax[i].plot(list_pd[1]["X"], list_pd[1]["Y"], color=color_cm.get_cmap('Spectral')[i]) 
    ax[i].plot(_x, _y, "*")
    ax[i].plot(list_pd[i]["X"], list_pd[i]["Y"], color=color[i]) 
    ax[i].set_title("Learning_Rate:" r'$\eta$'"="f"{etas[i]}")
    ax[3].plot(_x, _y, "*")
    ax[3].plot(list_pd[i]["X"], list_pd[i]["Y"], color=color[i])
    ax[3].set_title("Learning_Rate:" r'$\eta$'"="f"{etas[0]}, {etas[1]}, {etas[2]}")
plt.show()
   

#Stochastic GD
n_epochs = 50; t0, t1 = 5, 50; # learning schadule hyperparameters
def learning_schedule(t):
    return t0/(t + t1)

theta = np.random.randn(2, 1)   #random initialization
tmp_theta = []
for epoch in range(n_epochs):   #We have 50 epoch
    _theta = []
    for i in range(m):
        random_index = np.random.randint(m)
        xi = x_b[random_index:random_index + 1]
        yi = _y[random_index:random_index + 1]
        gradients = 2 * xi.T.dot(xi.dot(theta) - yi)
        eta = learning_schedule(epoch * m + i)
        theta = theta - eta * gradients
        _theta.append(theta)
    tmp_theta.append(theta)
plt.figure()
plt.plot(np.array(list(range(n_epochs))), np.array(tmp_theta)[:, 0, 0], "*")
# around 20 epoch it reached convergent, shuffle is useful ???????
plt.plot([20, 20],[4.15, 4.6], "--") 
        
#mini batch GD merriege between Batch or full GD with Stochastic GD


#life is not simple instead it is complex which is beautiful. 
#Now it is time talk Poly regression
m = 100; X = 6 * np.random.rand(m, 1) - 3
y = 0.5 * X**2 + X + 2 + np.random.rand(m, 1) 
from sklearn.preprocessing import PolynomialFeatures
poly_features = PolynomialFeatures(degree=2, include_bias=False)
X_poly = poly_features.fit_transform(X)

lin_reg = LinearRegression()
lin_reg.fit(X_poly, y)

print(lin_reg.intercept_, "----------------", lin_reg.coef_)
"""
How do we plot the data ?????????????????
"""

#learn the curvez, Regularized Linear Model, Ridge Regression and 
#Lasso regression or just both - Elestic Regrassion
#over fitted or underfitted or you need higher degree of poly, for identify this 
#we need train our data

from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split

def plot_learning_curves(model, X, y):
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2)
    train_errors, val_errors = [], []
    for m in range(1, len(X_train)):
        model.fit(X_train[:m], y_train[:m])
        y_train_predict = model.predict(X_train[:m])
        y_val_predict = model.predict(X_val)
        train_errors.append(mean_squared_error(y_train[:m], y_train_predict))
        val_errors.append(mean_squared_error(y_val, y_val_predict))
    plt.figure() 
    plt.xlabel("Training set size")
    plt.ylabel("RMSE")
    plt.plot(np.sqrt(train_errors), "r-+", linewidth=2, label="train")
    plt.plot(np.sqrt(val_errors), "b*", linewidth=3, label="val")
   
lin_reg = LinearRegression()
plot_learning_curves(lin_reg, X, y)
        

# from sklearn.pipeline import Pipeline

# polynomial_regression = Pipeline(["poly_features", PolynomialFeatures(degree=10, include_bias=False), ("lin_reg", LinearRegression()),])

# plot_learning_curves(polynomial_regression, X, y)       
        

#How to choose model using cross validation function get score

        
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split

digits = load_digits()

X_train, X_test, y_train, y_test = train_test_split(digits.data, digits.target, test_size=0.3)
#I am going to use logistic regression
LR = LogisticRegression()
LR.fit(X_train, y_train)
LR.score(X_test, y_test)

#SVM
svm = SVC()
svm.fit(X_train, y_train)
svm.score(X_test, y_test)
        
# Random Forst RF       

RF = RandomForestClassifier(n_estimators=40)
RF.fit(X_train, y_train)
RF.score(X_test, y_test)

#Now it is time use internal function: Cross Validation function which has real power make life simple

from sklearn.model_selection import cross_val_score
cross_val_score(LogisticRegression(), digits.data, digits.target)
cross_val_score(SVC(n_estimators=40), digits.data, digits.target)
cross_val_score(RandomForestClassifier(n_estimators=40), digits.data, digits.target)




        
        
"""
Real example for Boston house price evelution and presiction of it
"""

from sklearn import datasets
from sklearn.linear_model import ElasticNet

# Loading pre-defined Boston Dataset
# load dataset
url = 'https://raw.githubusercontent.com/jbrownlee/Datasets/master/housing.csv'
url1 = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/housing.names"
dataframe = pd.read_csv(url, header=None)
data = dataframe.values
#data.shape or dataframe.shape can print dimention of the data set((506, 14) ,(506, 13))
# X (506, 12) data while y (506, )
X, y = data[:, :-1], data[:, -1]

# define model
model = ElasticNet(alpha=1.0, l1_ratio=0.5)


#☺Logistic regression: Decision Boundaries; using Iris data set from datasets
#It is from Aurolien's books page 146
iris = datasets.load_iris()
X = iris["data"][:, 3:] #petal width
y = (iris["target"]==2).astype(np.int) # from this iris["target"]==2 get True or False than convert to 0 or 1. 1 If Iris verginica, esle 0; Very simple and useful!!

log_reg = LogisticRegression()
log_reg.fit(X, y)
X_new = np.linspace(0, 3, 1000).reshape(-1, 1); y_proba = log_reg.predict_log_proba(X_new)
plt.plot(X_new, y_proba[:, 1], "g-", label="Iris versigica")
plt.plot(X_new, y_proba[:, 0], "b--", label="Not Iris verginica")


y_proba = log_reg.predict_log_proba(X)
plt.plot(X, y_proba[:, 1], "g-", label="Iris versigica")
plt.plot(X, y_proba[:, 0], "b--", label="Not Iris verginica")




print(iris.data[:5])
print(iris.feature_names)

print(iris.target[:10])
print(iris.target_names)


# a) Python Libraries for LogisticRegression
# b) Built-in Iris Dataset

sbn.set_theme(style="darkgrid")

df = pd.DataFrame(iris.data, columns=iris.feature_names)
sbn.pairplot(df)

# d) Creating Train Test Split via train_test_split

X_train, X_test, y_train, y_test = train_test_split(iris.data, iris.target, test_size=0.2)

        
# d) Creating Logistic Regression Classifier

log_reg_02=LogisticRegression(C=0.02)

# e) Training Logistic Regression

log_reg_02.fit(X_train, y_train)

# f) Predicting with Logistic Regression

yhat = log_reg_02.predict(X_test)

print (yhat [0:5])
print (y_test [0:5])
        
from sklearn import preprocessing

#convert y values to categorical values
lab = preprocessing.LabelEncoder()
y_transformed = lab.fit_transform(y.reshape(-1))        
        
lst = [10, 5, 1, 0.1, 0.01, 0.001, 0.0001]

plt.figure(figsize=(12,12))
y_transformed = lab.fit_transform(y.reshape(-1))
for i, j in zip(range(6), lst):
    # clf = LogisticRegression(C=j)
    clf = LogisticRegression(C=j).fit(X, y_transformed)
    # clf.fit(X, y_transformed) 
    
    X1min, X1max = X[:, 0].min() - 1, X[:, 0].max() + 1
    X2min, X2max = X[:, 1].min() - 1, X[:, 1].max() + 1
    step = 0.01
    X1, X2 = np.meshgrid(np.arange(X1min, X1max, step), 
                          np.arange(X2min, X2max, step))
    
    Z = clf.predict(np.c_[X1.ravel(), X2.ravel()])
    Z = Z.reshape(X1.shape)
    
    cmap_dots = ['lightblue', 'brown', 'green']
    # plt.figure(dpi=800)
    ax = plt.subplot(3,2,i+1)
    # plt.figure(dpi=800)
    plt.title('Logistic Regression, C:{}'.format(j), size=8); plt.xticks(()); plt.yticks(())
    # plot_decision_regions(X_train2, y_train, clf=LogR, legend=2, colors=colors)
    
    # plt.figure(figsize=(8, 6))
    plt.contourf(X1, X2, Z, cmap="binary")
    
    sbn.scatterplot(x=X[:, 0], y=X[:, 1], hue=data.target_names[y],
                    palette=cmap_dots, alpha=0.9, edgecolor="black",)    
    plt.legend(loc='upper left')
        
        
        







df = pd.read_csv("Advertising.csv")

# fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(16, 6))

# axes[0].plot(df["TV"], df["sales"], "o")
# axes[0].set_ylabel("sales")
# axes[0].set_title("TV Spend")

# axes[1].plot(df["radio"], df["sales"], "o")
# axes[1].set_ylabel("Sales")
# axes[1].set_title("Radio spend")

# axes[2].plot(df["newspaper"], df["sales"], "o")
# axes[2].set_ylabel("sales")
# axes[2].set_title("Newspaper Spend")
# plt.tight_layout()

# sbn.pairplot(df)
# X = df.drop(columns="sales")
# y = df["sales"]

# #Here I don't have any idea shaffle(Bool) gives ordered or ramdon selection
# #in our case it is random
# #X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3,shaffle=True, False random_state=101)

# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=101)

# #30% of test data and 70% of training data
# print(len(X_train), len(X_test), len(y_train), len(y_test))

# #Linear Regression Y = ax + b

# LRmodel = LinearRegression()


# LRmodel.predict(X_test)
# ytest_predict= LRmodel.predict(y_test)
# LRmodel.fit(X_train, y_train)

# from sklearn.metrics import mean_absolute_error, mean_squared_error
# print("Average sales:", df["sales"].mean())

# sbn.histplot(data=df, x="sales", bins=20)
# mean_absolute_error(y_test, ytest_predict)



