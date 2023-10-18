#preprocessing the dataset
# Naive Bayes Classification
# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns; 

#Importing the dataset
dataset = pd.read_csv('creditcard.csv')
X = dataset.iloc[:, :-1]
y = dataset.iloc[:, 30]

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 1/3, random_state = 2)


# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# Fitting K-NN to the Training set
from sklearn.neighbors import KNeighborsClassifier
classifier = KNeighborsClassifier(n_neighbors = 5, metric = 'minkowski', p = 2)
classifier.fit(X_train, y_train)

# Predicting the Training set results
y_pred = classifier.predict(X_train.values)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_train, y_pred)

#Heat map of a confusion matrix
import seaborn as sns
sns.heatmap(cm,fmt=".0f",xticklabels=['CreditcardFraud_No','CreditcardFraud_Yes'],yticklabels=['CreditcardFraud_No','CreditcardFraud_Yes'],annot=True)
#sns.heatmap(cm,fmt=".0f",annot=True)

#Calculating Performance Metrics for Training Set
FP = cm.sum(axis=0) - np.diag(cm) 
FN = cm.sum(axis=1) - np.diag(cm)
TP = np.diag(cm)
TN = cm.sum() - (FP + FN + TP)
FP = FP.astype(float)
FN = FN.astype(float)
TP = TP.astype(float)
TN = TN.astype(float)
# Sensitivity, hit rate, recall, or true positive rate
TPR = TP/(TP+FN)
print("Recall",TPR)
# Specificity or true negative rate
TNR = TN/(TN+FP) 
print("Specificity",TNR)
# Precision or positive predictive value
PPV = TP/(TP+FP)
print("Precision",PPV)
# Negative predictive value
NPV = TN/(TN+FN)
print("Negative Predictive Value",NPV)
# Fall out or false positive rate
FPR = FP/(FP+TN)
print("False Positive Rate",FPR)
# False negative rate
FNR = FN/(TP+FN)
print("False Negative Rate",FNR)
# False discovery rate
FDR = FP/(TP+FP)
print("False Discovery Rate",FDR)
# Overall accuracy for each class
ACC = (TP+TN)/(TP+FP+FN+TN)
print("Accuracry",ACC)


# Fitting K-NN to the Testing set
from sklearn.neighbors import KNeighborsClassifier
classifier = KNeighborsClassifier(n_neighbors = 5, metric = 'minkowski', p = 2)
classifier.fit(X_test, y_test)

# Predicting the Testing set results
y_pred1 = classifier.predict(X_test)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm1 = confusion_matrix(y_test, y_pred1)

#create an empty data frame that we have to predict
variety=pd.DataFrame()
variety['Time']=[0]
variety['V1']=[-1.3598071336738]
variety['V2']=[-0.0727811733098497]
variety['V3']=[2.53634673796914]
variety['V4']=[1.37815522427443]
variety['V5']=[-0.338320769942518]
variety['V6']=[0.462387777762292]
variety['V7']=[0.239598554061257]
variety['V8']=[0.0986979012610507]
variety['V9']=[0.363786969611213]
variety['V10']=[0.0907941719789316]
variety['V11']=[-0.551599533260813]
variety['V12']=[-0.617800855762348]
variety['V13']=[-0.991389847235408]
variety['V14']=[-0.311169353699879]
variety['V15']=[1.46817697209427]
variety['V16']=[-0.470400525259478]
variety['V17']=[0.207971241929242]
variety['V18']=[0.0257905801985591]
variety['V19']=[0.403992960255733]
variety['V20']=[0.251412098239705]
variety['V21']=[-0.018306777944153]
variety['V22']=[0.277837575558899]
variety['V23']=[-0.110473910188767]
variety['V24']=[0.0669280749146731]
variety['V25']=[0.128539358273528]
variety['V26']=[-0.189114843888824]
variety['V27']=[0.133558376740387]
variety['V28']=[-0.0210530534538215]
variety['Amount']=[149.62]
print(variety)

y_pred1=classifier.predict(variety.values)
print("Did this transcation is Fraud:")
print(y_pred1)



