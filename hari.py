#Our Aim to detect someone have diabetes or not



#Import the libraries
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import VotingClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_curve, roc_auc_score
import matplotlib
import matplotlib.pyplot as plt



from PIL import Image
import streamlit as st



#Create title and sub
st.write("""
# Diabetes Detection
Detect if someone has diabetes using ML @ Python
""")



#Open and Display cover image
image = Image.open('C:/Users/User/Desktop/ML Project/posterr.jpg')
st.image(image, caption='ML', use_column_width=True)



#Get the data
df = pd.read_csv('C:/Users/User/Desktop/ML Project/diabetes.csv')



#Set subheader
st.subheader('Data Information:')



#Show the data as a table
st.dataframe(df)



#Show statistic on the data
st.write(df.describe())



#Show the data as a chart
chart = st.bar_chart(df)



#Split data into independent 'X' and dependent 'Y' variables



#selecting first to eighth column
X =df.iloc[:, 0:8].values



#selecting the last column
Y =df.iloc[:, -1].values



#Split data into 70% training and 30% testing
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.30, random_state=0)



##Get the feature input from user
def get_user_input():
    pregnancies = st.sidebar.slider('pregnancies', 0, 17, 3)
    glucose = st.sidebar.slider('glucose', 0, 199, 117)
    blood_pressure = st.sidebar.slider('blood_pressure', 0, 122, 72)
    skin_thickness = st.sidebar.slider('skin_thickness', 0, 99, 23)
    insulin = st.sidebar.slider('insulin', 0.0, 846.0, 30.0)
    BMI = st.sidebar.slider('BMI', 0.0, 67.1, 32.0)
    DPF = st.sidebar.slider('DPF', 0.078, 2.42, 0.3725)
    age = st.sidebar.slider('age', 21, 81, 29)
    
    #Store dictionary into a variable
    user_data = {'pregnancies': pregnancies,
                 'glucose': glucose,
                 'blood_pressure': blood_pressure,
                 'skin_thickness': skin_thickness,
                 'insulin':insulin,
                 'BMI': BMI,
                 'DPF': DPF,
                 'age': age
                 }
    #Transform the data into data frame
    features = pd.DataFrame(user_data, index = [0])
    return features

#Store the user input into variable
user_input = get_user_input()



#Set a subheader and display users input
st.subheader('User Input:')
st.write(user_input)



#Create and train the model
RandomForestClassifier = RandomForestClassifier()
RandomForestClassifier.fit(X_train, Y_train)



GaussianNB = GaussianNB()
GaussianNB.fit(X_train,Y_train)









#Show models metrics



st.subheader('Model Test Accuracy Score for Gaussian Naive Bayes Classifier:')
st.write( str(accuracy_score(Y_test, GaussianNB.predict(X_test)) * 100)+'%' )



st.subheader('Model Test Accuracy Score for Random Forest Classifier:')
st.write( str(accuracy_score(Y_test, RandomForestClassifier.predict(X_test)) * 100)+'%' )



ns_probs = [0 for _ in range(len(Y_test))]
lr_probs = RandomForestClassifier.predict_proba(X_test)
# keep probabilities for the positive outcome only
lr_probs = lr_probs[:, 1]
# calculate scores
ns_auc = roc_auc_score(Y_test, ns_probs)
lr_auc = roc_auc_score(Y_test, lr_probs)
# summarize scores
print('No Skill: ROC AUC=%.3f' % (ns_auc))
print('Logistic: ROC AUC=%.3f' % (lr_auc))
# calculate roc curves
ns_fpr, ns_tpr, _ = roc_curve(Y_test, ns_probs)
lr_fpr, lr_tpr, _ = roc_curve(Y_test, lr_probs)
# plot the roc curve for the model
plt.plot(ns_fpr, ns_tpr, linestyle='--', label='No Skill')
plt.plot(lr_fpr, lr_tpr, marker='.', label='Logistic')
# axis labels
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
# show the legend
plt.legend()
# show the plot
plt.savefig('demo.png', bbox_inches='tight')
#Open and Display cover image
image = Image.open('C:/Users/User/Desktop/ML Project/demo.png')
st.image(image, use_column_width=True)




#Store the models predictions in a variables




#Store the models predictions in a variables
prediction = GaussianNB.predict(user_input)



#Set a subheader and display classification
st.subheader('Classification: ')
st.write(prediction)