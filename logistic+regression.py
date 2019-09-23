import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
 
from sklearn.linear_model import LogisticRegression
from sklearn.externals import joblib
 
# Read CSV
insuranceDF = pd.read_csv('insurance.csv')
print(insuranceDF.head())
 
insuranceDF.info()
 
corr = insuranceDF.corr()
 
# Train Test Split
dfTrain = insuranceDF[:1000]
dfTest = insuranceDF[1000:1300]
dfCheck = insuranceDF[1300:]
 
# Convert to numpy array
trainLabel = np.asarray(dfTrain['insuranceclaim'])
trainData = np.asarray(dfTrain.drop('insuranceclaim',1))
testLabel = np.asarray(dfTest['insuranceclaim'])
testData = np.asarray(dfTest.drop('insuranceclaim',1))
 
# Normalize Data
means = np.mean(trainData, axis=0)
stds = np.std(trainData, axis=0)
trainData = (trainData - means)/stds
testData = (testData - means)/stds
 
insuranceCheck = LogisticRegression()
print(insuranceCheck.fit(trainData, trainLabel))
 
# Now use our test data to find out accuracy of the model.
 
accuracy = insuranceCheck.score(testData, testLabel)
print("accuracy = ", accuracy * 100, "%")