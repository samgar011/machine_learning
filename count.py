import csv 

import MySQLdb

mydb = MySQLdb.connect(host="127.0.0.1", user="root", password="", database="logitregression_data")

with open('test.csv', 'r') as file:
    reader = csv.reader(file, delimiter=',')
    all_value=[]
    sample_value['1','2','3']
    for row in reader:
        value = (row[0], row[1], row[2], row[3],row[4], row[5], row[6], row[7])
        all_value.append(value)

query = "INSERT INTO `BASELINE`(`ID`, `TWEETS`, `TOKENIZED`, `STOP_WORDS`, `STEMMED`, `POLARITY`, `SUBJECTIVITY`, `SENTIMENT`) VALUES (%s,%s,%s,%s,%s,%s,%s,%s)"

mycursor = mydb.cursor()
mycursor.executemany(query, all_value)
mydb.commit()