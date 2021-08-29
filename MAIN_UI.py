import PyQt5
from PyQt5 import QtWidgets,QtCore, QtGui
from PyQt5.QtWidgets import QApplication, QWidget, QPushButton, QLineEdit, QLabel, QTextEdit, QGroupBox, QMessageBox, QVBoxLayout, QSizePolicy,QFileDialog,QTableWidget,QTableWidgetItem
from PyQt5.QtGui import QIcon
from PyQt5.QtCore import pyqtSlot
import pandas as pd
import numpy as np
import matplotlib,csv
matplotlib.use('Qt5Agg')
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg
from matplotlib.figure import Figure
import pathlib
import re


#-------------SETTING THE GRAPH FOR OLD ALGORITHM-----------------
class MplCanvas(FigureCanvasQTAgg):
    
    def __init__(self, parent=None, width=5, height=4, dpi=100):
        fig = Figure(figsize=(width, height), dpi=dpi)
        self.axes = fig.add_subplot(111)
        super(MplCanvas, self).__init__(fig)


#-------------SETTING THE GRAPH FOR NEW ALGORITHM-----------------
class newMplCanvas(FigureCanvasQTAgg):
    
    def __init__(self, parent=None, width=5, height=4, dpi=100):
        fig = Figure(figsize=(width, height), dpi=dpi)
        self.axes = fig.add_subplot(111)
        super(newMplCanvas, self).__init__(fig)







# It is considered a good tone to name classes in CamelCase.
class MyFirstGUI(QtWidgets.QDialog): 
    file_select = ''
    def __init__(self):
        
        # Initializing QDialog and locking the size at a certain value
        super(MyFirstGUI, self).__init__()
        self.setFixedSize(1200, 670)
        self.fileName = "test.csv"
        self.model = QtGui.QStandardItemModel(self)
        self.setWindowTitle("HYBRID LOGISTIC REGRESSION ON TWITTER POST")


        #--------------DIVIDER--------------------
        self.divider_top = QGroupBox(self)
        self.divider_top.resize(1200,4)
        self.divider_top.move(1,75)


        self.divider_sideleft = QGroupBox(self)
        self.divider_sideleft.resize(4,500)
        self.divider_sideleft.move(500,75)


        self.divider_sideright = QGroupBox(self)
        self.divider_sideright.resize(4,500)
        self.divider_sideright.move(700,75)

        #---------------BUTTONS AND TEXT-----------------------------
    
        self.lbl_test = QLabel('TEST_SIZE', self)
        self.lbl_test.move(30,30)
        self.txt_test = QLineEdit(self)
        self.txt_test.move(100,30)



        self.sentiment = QPushButton('SENTIMENT', self)
        self.sentiment.move(700,20)
        self.sentiment.resize(100,50)
        self.sentiment.clicked.connect(self.senti)

        self.generate = QPushButton('GENERATE', self)
        self.generate.setToolTip('Generate The Algorithm')
        self.generate.move(820,20)
        self.generate.resize(100,50)
        #self.generate.setEnabled(False)
        self.generate.clicked.connect(self.on_click)

        self.clear = QPushButton('CLEAR', self)
        self.clear.move(940,20)
        self.clear.resize(100,50)
        self.clear.setEnabled(False)
        self.clear.clicked.connect(self.on_clear)

        self.exitui = QPushButton('EXIT', self)
        self.exitui.move(1050,20)
        self.exitui.resize(100,50)
        self.exitui.clicked.connect(self.terminate)

        


        


        
        #----GRAPH BOX FOR OLD ALGORITHM--------
        self.lbl_numb = QLabel('OLD ALGORITHM', self)
        self.lbl_numb.move(150,100)
        self.oldalgo_GraphBox = QGroupBox(self)
        self.oldalgo_GraphBox.resize(350,250)
        self.oldalgo_GraphBox.move(30,120)

        self.lbl_accurateold = QLabel(self)
        self.lbl_accurateold.setText('ACCURACY: ')
        self.lbl_accurateold.move(70,380)


        self.lbl_confuseold = QLabel(self)
        self.lbl_confuseold.setText('CONFUSION MATRIX')
        self.lbl_confuseold.move(390,200)


        


        #-------------------------------------------------


        #----GRAPH BOX FOR NEW ALGORITHM--------
        self.newalgo_GraphBox = QtWidgets.QGroupBox('', self)
        self.newalgo_GraphBox.resize(350,250)
        self.newalgo_GraphBox.move(830,120)

        self.lbl_numb = QLabel(self)
        self.lbl_numb.setText('NEW ALGO')
        self.lbl_numb.move(950,100)

        self.lbl_accuratenew = QLabel(self)
        self.lbl_accuratenew.setText('ACCURACY: ')
        self.lbl_accuratenew.move(850,380)


        self.lbl_confusenew = QLabel(self)
        self.lbl_confusenew.setText('CONFUSION MATRIX')
        self.lbl_confusenew.move(720,200)


        

        #--------------------RESULT FOR OLD ALGO------------------------
        

        self.lbl_tally_positive = QLabel(self)
        self.lbl_tally_positive.setText('POSITIVE: ')
        self.lbl_tally_positive.move(50,430)

        self.lbl_tally_neutral = QLabel(self)
        self.lbl_tally_neutral.setText('NEUTRAL: ')
        self.lbl_tally_neutral.move(180,430)

        self.lbl_tally_negative = QLabel(self)
        self.lbl_tally_negative.setText('NEGATIVE: ')
        self.lbl_tally_negative.move(280,430)

        self.lbl_tally_overall = QLabel(self)
        self.lbl_tally_overall.setText('OVERALL RESULT: ')
        self.lbl_tally_overall.move(150,470)

        #--------------------RESULT FOR NEW ALGO------------------------
        

        self.lbl_tally_hybpositive = QLabel(self)
        self.lbl_tally_hybpositive.setText('POSITIVE: ')
        self.lbl_tally_hybpositive.move(800,430)

        self.lbl_tally_hybneutral = QLabel(self)
        self.lbl_tally_hybneutral.setText('NEUTRAL: ')
        self.lbl_tally_hybneutral.move(950,430)

        self.lbl_tally_hybnegative = QLabel(self)
        self.lbl_tally_hybnegative.setText('NEGATIVE: ')
        self.lbl_tally_hybnegative.move(1080,430)

        self.lbl_tally_hyboverall = QLabel(self)
        self.lbl_tally_hyboverall.setText('OVERALL RESULT: ')
        self.lbl_tally_hyboverall.move(940,470)


        #--------------------------------------------




        #----------------------TABLE FOR THE LIST OF GATHERED TWEETS-------------------
     

        self.tableWidget = QtWidgets.QTableWidget(self)
        self.tableWidget.resize(1100, 150)
        self.tableWidget.move(40,500)
        self.tableWidget.setColumnCount(14)
        self.tableWidget.setObjectName("tableWidget")
        self.tableWidget.setHorizontalHeaderLabels(["ID","TWEETS","TOKENIZED","STOP WORDS","STEMMED","POLARITY","SUBJECTIVITY","SENTIMENT","BASELINE VALUE","BASELINE SENTIMENT","BASELINE RESULT","HYBRID VALUE","HYBRID SENTIMENT", "HYBRID RESULT"])
        self.tableWidget.doubleClicked.connect(self.getSelectedItemData)
        self.tableWidget.setSelectionBehavior(self.tableWidget.SelectRows)
        
    
    def terminate(self):
        exit()

    def getSelectedItemData(self):
        paths = []
        selected = self.tableWidget.selectedItems()
        if selected:
            for item in selected:
                if item.column() == 1:
                    tweet = item.data(0)
                if item.column() == 5:
                    polar = item.data(0)
                if item.column() == 6:
                    subject = item.data(0)    
                if item.column() == 7:
                    sentim = item.data(0)
                if item.column() == 8:
                    base_val = item.data(0)
                if item.column() == 9:
                    base_sent = item.data(0)
                if item.column() == 10:
                    base_res = item.data(0)
                if item.column() == 11:
                    hyb_val = item.data(0)
                if item.column() == 12:
                    hyb_sent = item.data(0)
                if item.column() == 13:
                    hyb_res = item.data(0)     
                
        print(paths)
        msg = QMessageBox()
        msg.setIcon(QMessageBox.Information)
        message = 'SENTIMENT: ' + sentim

        result_message = 'BASELINE VALUE: ' + base_val + '\n' + 'BASELINE SENTIMENT: ' + base_sent + "\n" + 'BASELINE RESULT: ' + base_res + "\n" + "======================================" + "\n"+ 'HYBRID VALUE: ' + hyb_val + '\n' + 'HYBRID SENTIMENT: ' + hyb_sent + "\n" + 'HYBRID RESULT: ' + hyb_res 
        msg.setText(tweet)
        msg.setInformativeText(message)
        msg.setWindowTitle("LOGISTIC REGRESSION")
        msg.setDetailedText(result_message)
        msg.setStandardButtons(QMessageBox.Ok | QMessageBox.Cancel)
            
        retval = msg.exec_()
        #print ("value of pressed message box button:", retval)




#-------------------GENERATE SENTIMENT----------------------------------
    def senti(self,event):
        try:
            self.file_select= self.openFileNameDialog(self)
            self.file_train = self.file_select
            self.getfile = self.file_train
          
            

            if self.file_train == "no file":
                pass
            else:
                from PREPREOCESS_SENTIMENT import Sentiment_process
                import PREPREOCESS_SENTIMENT
                self.pass_message = PREPREOCESS_SENTIMENT.Sentiment_process(self.getfile)
                
                self.sentiment_result = QMessageBox()
                self.sentiment_result.setWindowTitle("LOGISTIC REGRESSION")
                self.sentiment_result.setText("SENTIMENT TWEETS COMPLETE")
                self.sentiment_result.setIcon(QMessageBox.Information)
                self.y = self.sentiment_result.exec_()
                
            

        except Exception as e:
            print(e)
            
            self.result = QMessageBox()
            self.result.setWindowTitle("LOGISTIC REGRESSION")
            self.result.setText('ERROR')
            self.result.setIcon(QMessageBox.Warning)
            self.x = self.result.exec_()









#---------------GENERATING THE ALGORITHM-----------------------------------
    def on_click(self, event):
        try:
            self.file_select= self.openFileNameDialog(self)
            self.file_cv = self.file_select
            self.getfile = self.file_cv
            file_check = pathlib.Path(self.file_cv)
            if self.txt_test.text() == "":
                self.result = QMessageBox()
                self.result.setWindowTitle("LOGISTIC REGRESSION")
                self.result.setText('PLEASE PUT THE VALUE OF THE TEST SIZE')
                self.result.setIcon(QMessageBox.Warning)
                self.x = self.result.exec_()
            else:
                self.test_size = self.txt_test.text()
                self.NoOftest = float(self.test_size)
                self.test_num = self.NoOftest
                if file_check.exists():
                    if self.NoOftest >0.90:
                        self.result = QMessageBox()
                        self.result.setWindowTitle("LOGISTIC REGRESSION")
                        self.result.setText('PLEASE LIMIT THE TEST SIZE VALUE NOT MORE THAN 0.90 AND EQUAL TO 0.0')
                        self.result.setIcon(QMessageBox.Warning)
                        self.x = self.result.exec_()
                    
                    else:    
                        from BASELINE import baseline_algo
                        import BASELINE 
                        self.pass_message = BASELINE.baseline_algo(self.getfile, self.test_num)
                        #-----------------BASELINE RESULTS------------------------
                    
                        final_accurate = str(BASELINE.percentage)
                        final_pos = str(BASELINE.posi)
                        final_neu = str(BASELINE.neut)
                        final_neg = str(BASELINE.nega)
                        final_cm  = str(BASELINE.confuse)
                        final_report = str(BASELINE.reports)

                        post_accurate = 'ACCURACY: ' +  final_accurate
                        post_pos = 'POSITIVE: ' + final_pos
                        post_neu = 'NEUTRAL: ' + final_neu
                        post_neg = 'NEGATIVE: ' + final_neg
                        post_overall = 'OVERALL RESULT: ' + BASELINE.overall
                        post_cm = 'CONFUSION MATRIX' + '\n' + final_cm
                        #-----PLOT GRAPH FOR OLD-----------------
                        ploted = BASELINE.plots
                        self.m = MplCanvas(self, width=5, height=4)
                        self.m.axes.plot(ploted)
                        self.layout_oldalgo = QtWidgets.QVBoxLayout()
                        self.layout_oldalgo.addWidget(self.m)
                        self.oldalgo_GraphBox.setLayout(self.layout_oldalgo)
                        



                        self.lbl_accurateold.setText(post_accurate)
                        self.lbl_accurateold.adjustSize()

                        
                        
                        self.lbl_tally_positive.setText(post_pos)
                        self.lbl_tally_positive.adjustSize()

                        
                        self.lbl_tally_neutral.setText(post_neu)
                        self.lbl_tally_neutral.adjustSize()

                        
                        self.lbl_tally_negative.setText(post_neg)
                        self.lbl_tally_negative.adjustSize()

                        self.lbl_tally_overall.setText(post_overall)
                        self.lbl_tally_overall.adjustSize()


                        
                        from LOGSITC_REGRESSION_WITH_HYBRID import logistic_regression
                        import LOGSITC_REGRESSION_WITH_HYBRID
                        self.passed_message = LOGSITC_REGRESSION_WITH_HYBRID.logistic_regression(self.getfile,self.test_num)
                        #-----------------HYBRID RESULTS------------------------

                        final_hyb_accurate = str(LOGSITC_REGRESSION_WITH_HYBRID.percentage)
                        final_hyb_cm  = str(LOGSITC_REGRESSION_WITH_HYBRID.confuse)
                        final_hyb_report = str(LOGSITC_REGRESSION_WITH_HYBRID.reports)
                        post_hyb_accurate = 'ACCURACY: ' +  final_hyb_accurate
                        post_hyb_cm = 'CONFUSION MATRIX' + '\n' + final_hyb_cm 

                        final_hybpos = str(LOGSITC_REGRESSION_WITH_HYBRID.posi)
                        final_hybneu = str(LOGSITC_REGRESSION_WITH_HYBRID.neut)
                        final_hybneg = str(LOGSITC_REGRESSION_WITH_HYBRID.nega)
                        post_hybpos = 'POSITIVE: ' + final_hybpos
                        post_hybneu = 'NEUTRAL: ' + final_hybneu
                        post_hybneg = 'NEGATIVE: ' + final_hybneg
                        post_hyboverall = 'OVERALL RESULT: ' + LOGSITC_REGRESSION_WITH_HYBRID.overall

                        #----------------GRAPH FOR NEW-----------------------------
                        hyb_ploted = LOGSITC_REGRESSION_WITH_HYBRID.plots
                        self.n = newMplCanvas(self, width=5, height=4)
                        self.n.axes.plot(hyb_ploted)
                        self.layout_newalgo = QtWidgets.QVBoxLayout()
                        self.layout_newalgo.addWidget(self.n)
                        self.newalgo_GraphBox.setLayout(self.layout_newalgo)

                        
                        #------------------TEXT CHANGED WITH THE RESULT----------------------------
                    
                        self.lbl_accuratenew.setText(post_hyb_accurate)
                        self.lbl_accuratenew.adjustSize()

                    
                        #-------------HYBRID---------------------
                        self.lbl_tally_hybpositive.setText(post_hybpos)
                        self.lbl_tally_hybpositive.adjustSize()

                        
                        self.lbl_tally_hybneutral.setText(post_hybneu)
                        self.lbl_tally_hybneutral.adjustSize()

                        
                        self.lbl_tally_hybnegative.setText(post_hybneg)
                        self.lbl_tally_hybnegative.adjustSize()

                        self.lbl_tally_hyboverall.setText(post_hyboverall)
                        self.lbl_tally_hyboverall.adjustSize()

                        self.lbl_confuseold.setText(post_cm)
                        self.lbl_confuseold.adjustSize()

                        self.lbl_confusenew.setText(post_hyb_cm)
                        self.lbl_confusenew.adjustSize()

                        self.loaddata()
                        self.tableWidget.setHorizontalHeaderLabels(["ID","TWEETS","TOKENIZED","STOP WORDS","STEMMED","POLARITY","SUBJECTIVITY","SENTIMENT","BASELINE VALUE","BASELINE SENTIMENT","BASELINE RESULT","HYBRID VALUE","HYBRID SENTIMENT", "HYBRID RESULT"])
                        import os
                        if os.path.exists("temp_filebase.csv"):
                            os.remove("temp_filebase.csv")
                        else:
                            print("The file does not exist")
                        
                        
                        if os.path.exists("temp_file.csv"):
                            os.remove("temp_file.csv")
                        else:
                            print("The file does not exist")
        
        
                        #----------------MESSAGE BOX-------------------------
                        final_results = final_report + "\n" +"------------------------------" + "\n" + "WITH HYBRID" + "\n" +  final_hyb_report
                        self.result = QMessageBox()
                        self.result.setWindowTitle("LOGISTIC REGRESSION")
                        self.result.setText(final_results)
                        self.result.setIcon(QMessageBox.Information)
                        self.clear.setEnabled(True)
                        self.x = self.result.exec_()
                else:
                    self.result = QMessageBox()
                    self.result.setWindowTitle("LOGISTIC REGRESSION")
                    self.result.setText('FILE NOT FOUND, PLEASE CLICK THE SENTIMENT BUTTON TO CREATE THE TEST.CSV')
                    self.result.setIcon(QMessageBox.Information)
                    self.x = self.result.exec_()
        except Exception as e:
            print(e)
            
            self.result = QMessageBox()
            self.result.setWindowTitle("LOGISTIC REGRESSION")
            self.result.setText('ERROR')
            self.result.setIcon(QMessageBox.Warning)
            self.x = self.result.exec_()
        
    #-----------------FOR THE TABLE VIEW--------------------
    def loaddata(self):
        import MySQLdb
        
        mydb = MySQLdb.connect(host="127.0.0.1", user="root", password="", database="logitregression_data")
        mycursor = mydb.cursor()
        query = "SELECT baseline.ID, baseline.TWEETS,baseline.TOKENIZED,baseline.STOP_WORDS,baseline.STEMMED,baseline.POLARITY,baseline.SUBJECTIVITY, baseline.SENTIMENT,baseline_logitval.BASE_VALUE,baseline_logitval.BASE_SENTIMENT,baseline_logitval.BASE_RESULT, hybrid_logitval.HYB_VALUE,hybrid_logitval.HYB_SENTIMENT,hybrid_logitval.HYB_RESULT FROM baseline LEFT JOIN baseline_logitval on baseline.ID = baseline_logitval.BASE_ID LEFT JOIN hybrid_logitval ON baseline.ID = hybrid_logitval.HYB_ID WHERE baseline_logitval.BASE_VALUE IS NOT NULL"
        mycursor.execute(query)

        result = mycursor.fetchall()
        self.tableWidget.setRowCount(0)
        for row_number, row_data in enumerate(result):
            print(row_number)
            self.tableWidget.insertRow(row_number)
            for column_number, data in enumerate(row_data):
                #print(column_number)
                self.tableWidget.setItem(row_number, column_number, QTableWidgetItem(str(data)))

        
        

    


    def on_clear(self):

        #-----PLOT GRAPH FOR OLD-----------------
        self.m.setHidden(True) 
        #----------------GRAPH FOR NEW-----------------------------
        self.n.setHidden(True) 
        #------------------TEXT CHANGED WITH THE RESULT----------------------------
       
        self.lbl_accuratenew.setText('ACCURACY: ')
        self.lbl_accuratenew.adjustSize()

        self.lbl_accurateold.setText('ACCURACY: ')
        self.lbl_accurateold.adjustSize()

      

        self.lbl_tally_positive.setText('POSITIVE: ')
        self.lbl_tally_positive.adjustSize()

        
        self.lbl_tally_neutral.setText('NEUTRAL: ')
        self.lbl_tally_neutral.adjustSize()

        
        self.lbl_tally_negative.setText('NEGATIVE: ')
        self.lbl_tally_negative.adjustSize()

        self.lbl_tally_overall.setText('OVERALL RESULT: ')
        self.lbl_tally_overall.adjustSize()


        #-------------HYBRID--------------------
        
        self.lbl_tally_hybpositive.setText('POSITIVE: ')
        self.lbl_tally_hybpositive.adjustSize()

        

        self.lbl_tally_hybneutral.setText('NEUTRAL: ')
        self.lbl_tally_hybneutral.adjustSize()

        
        self.lbl_tally_hybnegative.setText('NEGATIVE: ')
        self.lbl_tally_hybnegative.adjustSize()

        self.lbl_tally_hyboverall.setText('OVERALL RESULT: ')
        self.lbl_tally_hyboverall.adjustSize()

        self.lbl_confuseold.setText('CONFUSION MATRIX')
        self.lbl_confuseold.adjustSize()

        self.lbl_confusenew.setText('CONFUSION MATRIX')
        self.lbl_confusenew.adjustSize()
        self.clear.setEnabled(False)
        self.layout_oldalgo.deleteLater()
        self.layout_newalgo.deleteLater()
        self.tableWidget.clear() 

        self.tableWidget.setHorizontalHeaderLabels(["ID","TWEETS","TOKENIZED","STOP WORDS","STEMMED","POLARITY","SUBJECTIVITY","SENTIMENT","BASELINE VALUE","BASELINE SENTIMENT","BASELINE RESULT","HYBRID VALUE","HYBRID SENTIMENT", "HYBRID RESULT"])

    #-------------------OPEN FILES FOR DATA------------------------

    def openFileNameDialog(self,event):
        options = QFileDialog.Options()
        options |= QFileDialog.DontUseNativeDialog
        fileName, _ = QFileDialog.getOpenFileName(self,"Open File", "","Excel Files (*.csv)", options=options)
        if fileName:
            self.file_select = fileName
            return self.file_select
        else:
            self.file_select = "no file"
            return self.file_select
    
        
            
if __name__ == '__main__':
    import sys
    app = QtWidgets.QApplication(sys.argv)
    gui = MyFirstGUI()
    gui.show()
    sys.exit(app.exec_())