from nltk.tokenize import sent_tokenize,word_tokenize
import tweepy
import nltk,re
import sys
import pandas as pd
#from MAIN_UI import *
import csv
from textblob import TextBlob 
import preprocessor as p
#from MAIN_UI import gathered

class sentimentanalyze():

    def sent_process(self):
        #gathered()
        consumerKey = ""
        consumerSecret = ""
        accessToken = ""
        accessTokenSecret = ""
        auth = tweepy.OAuthHandler(consumerKey, consumerSecret)
        auth.set_access_token(accessToken, accessTokenSecret)
        api = tweepy.API(auth)
        


        #tweetedid = []
        tweettexted = []
        



        #SEARCH THE FOLLOWING TERMS
        searchTerm = '#COVID'
        NoOfTerms = 200

        tweetsgath = api.search(q=searchTerm, count=NoOfTerms, lang = "en", include_rts = False)
     
        #-------DOWNLOAD AND CLEAN THE TWEETS--------
        
        with open('pretrain.csv', 'w',encoding='utf8', newline='') as file: 
            writer = csv.writer(file)
            writer.writerow(["ID", "TWEETS"])
           
            for tweet in tweetsgath:
                tweettexted.append(tweet.text)
                
            #-----------ONCE CLEARED, THEY WILL BE PUT IN A CSV FILE WITH AN ID---------   
        
 
            counter = 0
            for w in tweettexted:
                
                cleaned = p.clean(w)
                f_clean = cleaned.replace(':','')

                writer.writerow([counter, f_clean]) 
                counter+=1

        d = pd.read_csv(r'pretrain.csv', keep_default_na = False)
        d.drop_duplicates(subset = ["TWEETS"], inplace = True)
        print(d)
        d.to_csv('train.csv', index = False)
        

object = sentimentanalyze()
object.sent_process()
