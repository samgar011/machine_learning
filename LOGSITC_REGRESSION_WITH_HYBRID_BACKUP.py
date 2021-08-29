import sys, csv
#----------LOGISITC REGRESSION PART------------
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler 
from sklearn.metrics import accuracy_score
from numpy import log, dot, e
from numpy.random import rand
from sklearn.metrics import confusion_matrix, classification_report   

from sklearn.datasets import make_classification
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import GradientBoostingClassifier
from imblearn.over_sampling import RandomOverSampler
from imblearn.over_sampling import SMOTE
from collections import Counter
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import time

start = time.time()





class LogisticRegression():
    
    def sigmoid(self, z): return 1 / (1 + e**(-z))

    
    def softmax(self,x):
        e_x = np.exp(x - np.max(x))
        return e_x / e_x.sum(axis=0)
  
    def tanh(self,z):
        ez = np.exp(z)
        enz = np.exp(-z)
        return (ez - enz)/ (ez + enz)

    def cost_function(self, X, y, weights):                 
        z = dot(X, weights)
        predict_1 = y * log(self.sigmoid(z))
        predict_0 = (1 - y) * log(1 - self.sigmoid(z))
        return -sum(predict_1 + predict_0 ) / len(X)
    

    
    def fit(self, X, y, epochs=25, lr=0.05):        
        loss = []
        weights = rand(X.shape[1])
        N = len(X)
                 
        for _ in range(epochs):        
            # Gradient Descent
            y_hat = self.sigmoid(dot(X, weights))
            x_hat = self.softmax(dot(X, weights))
            z_hat = self.tanh(dot(X, weights))
            
            weights -= lr * dot(X.T,  y_hat - x_hat - z_hat- y) / N            
            # Saving Progress
            loss.append(self.cost_function(X, y, weights)) 
            
        self.weights = weights
        self.loss = loss
    
    
    def predict(self, X):        
        # Predicting with sigmoid,Softmax and tanh function
        z = dot(X, self.weights)
     
        soft = self.softmax(z)
        
        tand = self.tanh(soft)
        
    
        # Returning binary result
        x = []
        for i in tand:
            if i > 0.05:
                x.append(1)
                print("{:.2f}".format(i),': 1')    
            elif (i >= .01) and (i <=0.059):
                x.append(0)
                print("{:.2f}".format(i),": 0")
            else:
                x.append(-1)
                print("{:.2f}".format(i),': -1')
                
        return x




class logistic_regression():
    
    
    
    
    def __init__(self,getfile, test_num):
       
        #-----SPLIT DATASETS-------
        self.getfile = getfile
        self.test_num = test_num
        
        tested = pd.read_csv(self.getfile)
        x = tested.iloc[:, [5, 6]].values   
        # output 
        y = tested.iloc[:, 7].values
        xtrain, xtest, ytrain, ytest = train_test_split(x, y, test_size = self.test_num, random_state = 42) 

        #4 Feature Scaling


        #Feature Scaling or Standardization: It is a step of Data Pre Processing which is applied to independent variables or features of data. 
        # It basically helps to normalise the data within a particular range. Sometimes, it also helps in speeding up the calculations in an algorithm.


        sc_x = StandardScaler() 
        xtrain = sc_x.fit_transform(np.asarray(xtrain))  
        xtest = sc_x.transform(np.asarray(xtest))

        counter = Counter(y) 


      #---------------SMOTE ALGORITHM--------------------------
    
        print("Before OverSampling, counts of label '1': {}\n".format(sum(ytrain == 1))) 
        print("Before OverSampling, counts of label '-1': {} \n".format(sum(ytrain == -1))) 
        print('WITH SMOTE')

        os =  RandomOverSampler(sampling_strategy='minority')
        xtrain_res, ytrain_res = os.fit_sample(x, y)
        oversample = SMOTE()
        
        xtrain, ytrain = oversample.fit_resample(xtrain_res, ytrain_res.ravel())
                    
        counter = Counter(ytrain)
        print(counter)
        print('After OverSampling, the shape of train_X: {}'.format(xtrain.shape)) 
        print('After OverSampling, the shape of train_y: {} \n'.format(ytrain.shape)) 
    
        print("After OverSampling, counts of label '1': {}".format(sum(ytrain == 1))) 
        print("After OverSampling, counts of label '-1': {}".format(sum(ytrain == -1)))
        



















        #---------------LOGISTIC REGRESSION----------------------
        #5 Fitting the Logistic Regression to the Training Set: 
        #We create a classifier object of LR class
        
        classifier = LogisticRegression()


        #Fit logistic regression model to the training set (Xtrain and ytrain)
        classifier.fit(xtrain, ytrain)
       
    
        #6 Predicting the Test set results
        #Using predict method for the classifier object and put Xtest for #argument
        y_pred = classifier.predict(xtest)
        print(y_pred)
        posed = 1
        neued = 1
        neged = 1

        


        #-----------The Result On The Logistic Regression Process Based on the Number of Test size will be seperated and determine the overall Result--------------
        for over in y_pred:
          
            if over == 1:
                posed+=1
            elif over ==0:
                neued +=1     
            else: 
                neged+=1
        
    
        end = time.time()
        final_timed = end - start
        #---------------CONFUSION MATRIX----------------------
        #7 Making the Confusion Matrix. It contains the correct and incorrect predictions of our model 
       
        #ytest parameter will be y_test
        #y_pred is the logistic regression model prediction     
        cm = confusion_matrix(ytest, y_pred) 
        import warnings
        warnings.filterwarnings("ignore")
        cr = classification_report(ytest, y_pred)
        print(ytest)
        
        print ("Confusion Matrix : \n", cm)   
        print (cr)

        import mlxtend.plotting
        from mlxtend.plotting import plot_confusion_matrix
        class_names = ['-1', '0', '1']
        fig, ax = plot_confusion_matrix(conf_mat=cm,colorbar=True,class_names=class_names)
        fig.canvas.set_window_title('HYBRID LOGISTIC REGRESSION')
        plt.ylabel('Actual label')
        plt.xlabel('Predicted label')
        plt.show()


        
        







        #-------SENDS ALL VALUES TO APPEAR ON THE USER INTERFACE----------------
        global accurate, confuse,posi, neut, nega, overall,plots,replot,percentage, reports, final_time
        accurate = accuracy_score(ytest, y_pred)
        print(accurate)
        percentage = "{:.0%}".format(accurate)
        confuse = cm
        print(percentage)
        posi = posed
        neut = neued
        nega = neged
        plots = y_pred
        replot = plt
        reports = cr

       


        if (neut >= posi) and (neut >= nega):
            overall = 'NEUTRAL'
        elif (posi >= neut) and (posi >= nega):
            overall = 'POSITIVE'
        else:
            overall = 'NEGATIVE'

        
        print(overall)
        final_time = final_timed 
        
        #return percentage, confuse, posi, neut, nega, overall, plots, replot, reports, final_time           
                   
                

    
                
if __name__ == '__main__':
    logistic_regression()

