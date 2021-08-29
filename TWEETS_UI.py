import sys
import PyQt5
from PyQt5 import QtWidgets,QtCore, QtGui
from PyQt5.QtWidgets import QApplication, QLabel, QMainWindow,QWidget, QPushButton, QLineEdit, QTextEdit, QGroupBox, QMessageBox, QVBoxLayout, QSizePolicy
from PyQt5.QtCore import Qt

import tweepy
import nltk,re
import sys
import pandas as pd

import csv
from textblob import TextBlob 
import preprocessor as p
search = ''
NoOfTweets = 0
# Subclass QMainWindow to customise your application's main window
class MainWindow(QMainWindow):
    

    
    

    def __init__(self, *args, **kwargs):
        # Initializing QDialog and locking the size at a certain value
        super(MainWindow, self).__init__()
        self.setFixedSize(800, 400)
        self.fileName = "train.csv"
        self.model = QtGui.QStandardItemModel(self)
        self.setWindowTitle("HYBRID LOGISTIC REGRESSION ON TWITTER POST")


        #---------------BUTTONS-----------------------------
        self.getweet = QPushButton(self)
        self.getweet.setText('GET TWEETS')
        self.getweet.move(540,20)
        self.getweet.resize(100,50)
        self.getweet.clicked.connect(self.on_click)

        self.exitui = QPushButton('EXIT', self)
        self.exitui.move(650,20)
        self.exitui.resize(100,50)
        self.exitui.clicked.connect(self.terminate)

        #------------TEXTBOX-------------------------------
        self.lbl_search = QLabel('SEARCH', self)
        self.lbl_search.move(30,30)
        self.txt_search = QLineEdit(self)
        self.txt_search.move(100,30)
        

        self.lbl_num = QLabel('NUMBER OF TWEETS', self)
        self.lbl_num.move(250,30)
        self.txt_number = QLineEdit(self)
        self.txt_number.move(370,30)
        

        #----------------------TABLE FOR THE LIST OF GATHERED TWEETS-------------------
        self.tableView = QtWidgets.QTableView(self)
        self.tableView.setModel(self.model)
        self.tableView.resize(750, 150)
        self.tableView.move(20,200)
        #-------------------------------------------------------------------

    def on_click(self):
     
        consumerKey = "8mJXNmRuDyka7r1K0fZkE3mqw"
        consumerSecret = "TKO8JMbfpheWyNyNgs7FsNQpuJNBiZHBk8KMXkRn1lcfOmNh1p"
        accessToken = "1148024709746847744-GWvXY1V3T6FoU9iwm3OenKooh9IjAd"
        accessTokenSecret = "RsPAb6NvlNAki7PIG7jW1m4lsUTn3KUqoiTMdzG7Tfn2f"
        auth = tweepy.OAuthHandler(consumerKey, consumerSecret)
        auth.set_access_token(accessToken, accessTokenSecret)
        api = tweepy.API(auth)
        


        #tweetedid = []
        tweettexted = []
        search = self.txt_search.text()
        NoOfTweets = self.txt_number.text()
        import re
        searched = re.sub(r"([a-z\.!?])([A-Z])", r"\1 \2", search)
        if searched == '' or NoOfTweets == '':
        
        
            gathered_result = QMessageBox()
            gathered_result.setIcon(QMessageBox.Warning)
            gathered_result.setWindowTitle("TWEETS COLLECTOR")
            gathered_result.setText("EMPTY FIELD/S HAS BEEN DETECTED")
            gathered_result.setIcon(QMessageBox.Warning)
            y = gathered_result.exec_()

        else:
            #SEARCH THE FOLLOWING TERMS
            searchTerm = str(search)
            print(searchTerm)
            if searchTerm.count("#") >= 2 and NoOfTweets == '':
                gathered_result = QMessageBox()
                gathered_result.setWindowTitle("TWEETS COLLECTOR")
                gathered_result.setText("ONE HASHTAG OR TERM ONLY")
                gathered_result.setIcon(QMessageBox.Warning)
                y = gathered_result.exec_()
            elif searchTerm.count(' ') >=1:
                gathered_result = QMessageBox()
                gathered_result.setWindowTitle("TWEETS COLLECTOR")
                gathered_result.setText("ONE HASHTAG OR TERM ONLY")
                gathered_result.setIcon(QMessageBox.Warning)
                y = gathered_result.exec_()
            
            else:

                NoOfTerms = int(NoOfTweets)
                if NoOfTerms >= 1000:
                    gathered_result = QMessageBox()
                    gathered_result.setWindowTitle("TWEETS COLLECTOR")
                    gathered_result.setText("PLEASE LIMIT THE NUMBER OF TWEETS TO 1000 ONLY TO AVOID ERRORS")
                    gathered_result.setIcon(QMessageBox.Warning)
                    y = gathered_result.exec_()
                else: 
                    import urllib.request
                    host='http://google.com'
                   
                    
                    try:
                        urllib.request.urlopen(host)
                        print(NoOfTerms)
                        tweetsgath = tweepy.Cursor(api.search, q=searchTerm, lang = "en", include_entities=True, wait_on_rate_limit=True, include_rts = False).items(NoOfTerms)
                        #tweetsgath = api.search(q=searchTerm, count=NoOfTerms, lang = "en", include_rts = False)
                    
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

                        self.loadCsv(self.fileName)




                        gathered_result = QMessageBox()
                        gathered_result.setWindowTitle("TWEETS COLLECTOR")
                        gathered_result.setText("TWEETS HAS BEEN GATHERED")
                        gathered_result.setIcon(QMessageBox.Information)
                        y = gathered_result.exec_()
                      
                    except:
                        gathered_result = QMessageBox()
                        gathered_result.setWindowTitle("TWEETS COLLECTOR")
                        gathered_result.setText("No internet connection. Please Connect To The Internet")
                        gathered_result.setIcon(QMessageBox.Critical)
                        y = gathered_result.exec_()


    def terminate(self):
        sys.exit() 

     #-----------------FOR THE TABLE VIEW--------------------
    def loadCsv(self, fileName):
        self.model.clear()
        with open(fileName, "r", encoding='utf8') as fileInput:
            
            for row in csv.reader(fileInput):    
                items = [
                    QtGui.QStandardItem(field)
                    for field in row
                ]
                self.model.appendRow(items)

app = QApplication(sys.argv)

window = MainWindow()
window.show()

app.exec_()