import sys, csv
from textblob import TextBlob 
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

from sklearn.model_selection import train_test_split

from sklearn.preprocessing import StandardScaler 
from sklearn.metrics import accuracy_score
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
from threading import Thread
from functools import wraps
import re


from numpy import log, dot, e, where
from numpy.random import rand

class Sentiment_process():
    
    
    def __init__(self, getfile):
        try:
            
           
            polarity = 0
            col_list = ["ID", "TWEETS"]   
            self.getfile = getfile
            print(self.getfile)
            #-----READS THE DATASET OF THE TWEETS-----------
            df = pd.read_csv(self.getfile, usecols=col_list)
        
            reader = df["TWEETS"]
            #print(reader)
            with open('test.csv', 'w',encoding='utf8', newline='') as filed: 
                writer = csv.writer(filed)
                writer.writerow(["ID", "TWEETS","TOKENIZED","STOP WORDS", "STEMMED", "POLARITY", "SUBJECTIVITY", "SENTIMENT"])
                counter = 0
                tok_count = 0
                stop_wor_count = 0
                stem_count = 0
                tweetted = '' 
                for row in reader:
                    tweetted = row
                    #-----TOKENIZING TWEETS FROM THE TWEETS ROW-----------
                    textonly = re.sub("[^a-zA-Z]", " ",str(tweetted))
                    tokenized_text=sent_tokenize(textonly)
                    
                    for k in tokenized_text:
                        tokenized_sentence=sent_tokenize(k)
                        tok_count +=1
                    print(tok_count)   


                    filtered_sent=[]
                    for tokword in tokenized_sentence:
                        stop_words=set(stopwords.words('english'))
                        
                        new_sentence = ' '.join([word for word in tokword.split() if word not in stop_words])
                        filtered_sent.append(new_sentence)
                        stop_wor_count +=1
                    #print(stop_wor_count)
                
                        

                    
                #--------STEMMED TWEETS-----------------------
                    stemmed_tweet=[]
                    ps = PorterStemmer()
                    
                    for getsteam in filtered_sent:
                        getstoped = getsteam
                        stemmed_tweet.append(ps.stem(getsteam))
                        #print(stemmed_tweet)
                        stem_count +=1 
                    #print(stem_count)
                    

                    
                    
                    
                #--------------------GET SENTIMENT VALUE----------- 
                    for gather_steam in stemmed_tweet:
                        #steam_tweets = gather_steam
                        analysis = TextBlob(gather_steam)
                        counter+=1
                        sentied = 0
                        if analysis.sentiment.polarity > 0:
                            sentied = 1
                            
                        
                        else: 
                            
                            sentied = -1
                        
                        
                        #--------WRITE TO CSV FILE FOR SPLIT DATASETS-----------------------
                        writer.writerow([counter, tweetted, tokenized_sentence,getstoped, gather_steam, analysis.sentiment.polarity, analysis.sentiment.subjectivity, sentied])        
                    polarity += analysis.sentiment.polarity # adding up polarities to find the average later
            polarity = polarity / counter 
        except Exception as e:
    
           print(e)
                

if __name__ == '__main__':
    Sentiment_process

