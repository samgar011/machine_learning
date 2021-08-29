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



from numpy import log, dot, e, where
from numpy.random import rand



import time

start = time.time()
class LogisticRegression():
    
    def sigmoid(self, z): return 1 / (1 + e**(-z))
    
    def cost_function(self, X, y, weights):                 
        z = dot(X, weights)
        predict_1 = y * log(self.sigmoid(z))
        predict_0 = (1 - y) * log(1 - self.sigmoid(z))
        return -sum(predict_1 + predict_0) / len(X)
    
   
    
    def fit(self, X, y, epochs=25, lr=0.05):        
        loss = []
        weights = rand(X.shape[1])
        N = len(X)
                 
        for _ in range(epochs):        
            # Gradient Descent
            y_hat = self.sigmoid(dot(X, weights))
            weights -= lr * dot(X.T,  y_hat - y) / N            
            # Saving Progress
            loss.append(self.cost_function(X, y, weights)) 
            
        self.weights = weights
        self.loss = loss
    
    
    def predict(self, X):        
        # Predicting with sigmoid function
        z = dot(X, self.weights)
        # Returning binary result
        x = []
        #print (z)
        with open('temp_filebase.csv', 'w',encoding='utf8', newline='') as filed: 
            writer = csv.writer(filed)
            writer.writerow(["logit"])
            for i in z:
                if i > 0.0:
                    x.append(1)
                    file = str("{:.2f}".format(i))
                else:
                    x.append(0)
                    file = str("{:.2f}".format(i))
                writer.writerow([file])  

            return x

class baseline_algo():
    
    def __init__ (self):
        import LOGIT_HYB as hybs
        #-----SPLIT DATASETS-------
        tested = pd.read_csv('test.csv')
        x = tested.iloc[:, [5, 6]].values   
        # output 
        y = tested.iloc[:, 7].values
        xtrain, xtest, ytrain, ytest = train_test_split(x, y, test_size = 0.40, random_state = 42) 
      
        #4 Feature Scaling
        sc_x = StandardScaler() 
        xtrain = sc_x.fit_transform(np.asarray(xtrain))  
        xtest = sc_x.transform(np.asarray(xtest)) 
        
        
        counter = Counter(y)
        #---------------SMOTE ALGORITHM--------------------------
    
        print("Before OverSampling, counts of label '1': {}\n".format(sum(ytrain == 1))) 
        print("Before OverSampling, counts of label '-1': {} \n".format(sum(ytrain == 0))) 
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
        print("After OverSampling, counts of label '-1': {}".format(sum(ytrain == 0)))
        #---------------LOGISTIC REGRESSION----------------------

        #sklearn.linear_model LogisticRegression Interpretation
        #5 Fitting the Logistic Regression to the Training Set: 
    
        #There are many optional parameters. Lets only use random_state=0
        #We create a classifier object of LR class
        
        classifier = LogisticRegression()


        #Fit logistic regression model to the training set (Xtrain and ytrain)
        classifier.fit(xtrain, ytrain)
        
       
       
        #6 Predicting the Test set results
        #Using predict method for the classifier object and put Xtest for #argument
        y_pred = classifier.predict(xtest)
        #print(y_pred)
        posed = 0
        neged = 0




        #-----Access Database to store all data,since CSV is too hard to import all data in one---
        
        import MySQLdb
        
        mydb = MySQLdb.connect(host="127.0.0.1", user="root", password="", database="logitregression_data")
        mycursor = mydb.cursor()
        logit = []
        with open('temp_filebase.csv','r') as tempo:
            read = csv.reader(tempo,delimiter = ',')
            
            for tem in read:
                logit.append(tem)

        
            
            counter = 0
            mycursor.execute("DELETE FROM baseline_logitval")

            #-----------The Result On The Logistic Regression Process Based on the Number of Test size will be seperated and determine the overall Result--------------
            for over in y_pred:
                counter+=1
                if over == 1:
                    posed+=1
                    getov = over
                    #print("pos",getov)
                    resu = 'Positive'
                    regval  = 1
                else: 
                    neged+=1
                    getov = over
                    #print("neg",getov)
                    resu = 'Negative'
                    regval  = -1
                #stregval = str(regval)
                #valued = (counter,over,stregval, resu) 
                
                query2 = "INSERT INTO `baseline_logitval`(`BASE_ID`, `BASE_VALUE`, `BASE_SENTIMENT`, `BASE_RESULT`) VALUES (%s,%s,%s,%s)"
                mycursor.execute(query2, (counter, logit[counter],regval,resu))

           
        
        
        mydb.commit()  
        mydb.close()
       
        #---------------CONFUSION MATRIX----------------------
        #7 Making the Confusion Matrix. It contains the correct and incorrect predictions of our model 
       
        #ytest parameter will be y_test
        #y_pred is the logistic regression model prediction     
        cm = confusion_matrix(ytest, y_pred) 
        import warnings
        warnings.filterwarnings("ignore")
        cr = classification_report(ytest, y_pred)
        
        print ("Confusion Matrix : \n", cm)   
        print (cr)
        import mlxtend.plotting
        from mlxtend.plotting import plot_confusion_matrix
        class_names = ['-1','0', '1']
        fig, ax = plot_confusion_matrix(conf_mat=cm,colorbar=True,
                                class_names=class_names)
        fig.canvas.set_window_title('BASELINE LOGISTIC REGRESSION')
        plt.ylabel('Actual label')
        plt.xlabel('Predicted label')
        plt.show()




        
        #-----SENDS ALL VALUES TO APPEAR ON THE UI----------------
        global accurate, confuse,posi, neut, nega, overall,plots,replot,percentage, reports,final_time
        accurate = accuracy_score(ytest, y_pred)
        print(accurate)
        percentage = "{:.0%}".format(accurate)
        confuse = cm
        print(percentage)
        posi = posed
        
        nega = neged
        plots = y_pred
        replot = plt
        reports = cr

        

        
        if posi >= nega:
            overall = 'POSITIVE'
        else:
            overall = 'NEGATIVE'

        
        print(overall)
        #final_time = final_timed
        #print(final_time)
        #return percentage, confuse, posi, neut, nega, overall, plots, replot, reports, final_time           
        
                

    
                
if __name__ == '__main__':
    baseline_algo()



