
# CREDIT  CARD FRAUD DETECTION 

Credit card fraud detection is a set of methods and techniques designed to block fraudulent purchases, both online and in-store. This is done by ensuring that you are dealing with the right cardholder and that the purchase is legitimate.



## Acknowledgements

 - [Learn About ML using Python](https://www.geeksforgeeks.org/machine-learning-with-python)
 - [Explore the Dataset](https://www.kaggle.com/datasets/mlg-ulb/creditcardfrauds)
 - [Algorithm using in Project](https://www.geeksforgeeks.org/k-nearest-neighbors-with-python-ml)


## Documentation

[Documentation](https://en.wikipedia.org/wiki/Credit_card_fraud)



## Features
Selecting and engineering pertinent features from the dataset is necessary to create a credit card fraud detection model that works. Here's a step-by-step tutorial on selecting and preparing features for Python credit card fraud detection:


- **Import Libraries and Load Data:**

Importing the required libraries and loading your dataset of credit card transactions should come first. For this purpose, you can utilise libraries like scikit-learn, numpy, and pandas.

    import pandas as pd 
    import numpy as np  
    #Load your dataset 
    data = pd.read_csv("creditcard.csv")

- **Data Exploration:** 

Investigate your dataset to learn about its composition, feature distribution, and balance between legitimate and fraudulent transactions. For this, visualisation programmes like Seaborn and Matplotlib can be useful.



    import matplotlib.pyplot as plt 
    import seaborn as sns 
    #Explore the data

    print(data.head())
    print(data.info())
    print(data["Class"].value_counts())
    # Visualize data distributions
    sns.countplot(data["Class"])
     plt.show()

  - **Feature Selection**
  Determine which features are most important for detecting fraud. Univariate feature selection, feature importance from tree-based models, and domain expertise are a few popular methods.

- Feature selection using feature importance from a tree-based model .

     (e.g., Random Forest)

        from sklearn.ensemble import RandomForestClassifier

         X = data.drop("Class", axis=1)
         y = data["Class"]
        model = RandomForestClassifier()
      model.fit(X, y)
      feature_importance = model.feature_importances_ 
      features = X.columns

       # Select the top N important features
       N = 10
       selected_features = features[np.argsort(feature_importance)[::-1][:N]] 
       print("Selected Features:", selected_features)

- **Feature Engineering:**
Develop new features or preprocess current ones to improve the model's fraud detection capabilities. This may consist of
use StandardScaler or MinMaxScaler, for example, to scale numerical features to equivalent scales.
if any, encoding category features.
constructing aggregations or interaction characteristics to identify trends in the data.

        from sklearn.preprocessing import StandardScaler
        from sklearn.preprocessing import LabelEncoder
        #Scale numerical features
        scaler = StandardScaler()
        scaler = StandardScaler()scaler.fit_transform(X[selected_features])
        #Encode categorical features (if any)
        #For example, if you have a "category" column:
        #encoder = LabelEncoder()
        #X["category_encoded"] = encoder.fit_transform(X["category"])

  
- **Data Splitting:**

 Split the data into training and testing sets to train and evaluate your credit card fraud detection model. 

  

    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)


- **Modeling:**

  Using the features that have been carefully chosen and constructed, train a aud.machine learning or deep learning model that is suitable for detecting credit card fraud.



#Example using a Random Forest classifier

    from sklearn.ensemble import RandomForestClassifier

    model = RandomForestClassifier()
    model.fit(X_train, y_train)

    #Make predictions

    y_pred = model.predict(X_test)

**Model Evaluation:**

Assess the model's performance using relevant metrics, such as accuracy, precision, recall, F1-score, and area under the ROC curve (AUC-ROC).



    from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_auc_score 
    ("Classification Report:")

    print(classification_report(y_test, y_pred))

    print("Confusion Matrix:")

    print(confusion_matrix(y_test, y_pred))

    print("AUC-ROC Score:", roc_auc_score(y_test, y_pred))

Remember that feature engineering is an iterative process, and you may need to fine-tune your feature selection and engineering techniques based on your specific dataset and the performance of your model.

  

## Usage/Examples
- **K-Nearest Neighbors Algorithm**

```javascript
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




```

## Support

For support, email himanshukumar.cse2021@dscet.ac.in or join our [GitHub](https://github.com/himanshukr01).

## Authors

- [@Himanshu Kumar](https://github.com/himanshukr01)


  




    


     



  

  
     


 
