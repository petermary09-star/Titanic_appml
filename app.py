import streamlit as st
import numpy as np
import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score

#import joblib


# st.cache_resource
# def load_model():
#     with open('Titanic_model.pkl', 'rb') as f:
#         return pickle.load(f)
    
# Titanic_model = load_model()


import pandas as pd
#import seaborn as sns
import numpy as np

import warnings

warnings.filterwarnings('ignore')

titanic = pd.read_csv('titanic.csv')

# titanic

# titanic.info()

#### Data cleaning

titanic['age'].mean()
titanic['age'].fillna(titanic['age'].mean())
titanic['age'] = titanic['age'].fillna(titanic['age'].mean())

# titanic.info()

titanic['embarked'].isna()
index = titanic[titanic['embarked'].isna()].index

titanic.drop(index,inplace=True)

# titanic.info()

titanic.drop('deck',axis=1,inplace=True)

# titanic.info()

#### Features Selection

# sns.pairplot(titanic)

y = titanic['survived']

x = titanic.drop(['survived', 'who', 'adult_male', 'alive', 'alone'],axis=1)


#### Splitting data into training and testing

#from sklearn.model_selection import train_test_split

x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.25,random_state=42)

# x_train

#### Feature engineering

#from sklearn.preprocessing import LabelEncoder, OneHotEncoder

label = LabelEncoder()

print(titanic['class'])

x['class_label'] = label.fit_transform(titanic['class'])

# x.head()

onehot = OneHotEncoder()

onehot.fit_transform(titanic[[  'sex'  ]])

pd.get_dummies(titanic[['sex', 'embarked', 'embark_town']])

pd.get_dummies(x[['sex', 'embarked', 'embark_town']],dtype=int)

encoded = pd.get_dummies(x[['sex', 'embarked', 'embark_town']],dtype=int,drop_first=True)

# encoded

x = pd.concat([x,encoded],axis=1)

# x.head()

x.drop(  ['sex', 'class', 'embark_town', 'embarked'],  axis=1,inplace=True)

# x.head()

#### Splitting into training and testing

#from sklearn.model_selection import train_test_split

x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.25,random_state=42)

# x_train

#### Algorithm selection

#from sklearn.linear_model import LogisticRegression

Titanic_model = LogisticRegression()

Titanic_model.fit(x_train, y_train)

predictions = Titanic_model.predict(x_test)

# predictions

table = pd.DataFrame(predictions,columns=['Predicitons'])
y_test.reset_index(drop=True)
table['y_test'] = y_test.reset_index(drop=True)
# table

### evaluation

#from sklearn.metrics import confusion_matrix, classification_report, accuracy_score

confusion_matrix(y_test,predictions)

# sns.heatmap(confusion_matrix(y_test,predictions),annot=True)

#print(classification_report(y_test, predictions))

#accuracy_score(y_test,predictions)




st.title('Titanic Survival Prediction App')
st.write('This app will predict Titanic Accident Survival Rate')

st.subheader('Survived versus Sex, Age, Class, Fare, Embarked Town')

pclass = st.sidebar.select_slider('select your class between 123', options=[1,2,3])
age = st.sidebar.number_input('Age', 0, 100)
sibsp = st.sidebar.select_slider('select between 0 and 1', options=[0,1])
sex_male = st.sidebar.select_slider('sex; 0 female, 1 rep male',options=[0,1])
fare = st.sidebar.number_input('fare')
class_label = st.sidebar.select_slider('select class of 123',options=[1,2,3])
parch = st.sidebar.select_slider('select class of 012',options=[0,1,2])
embarked_Q = st.sidebar.select_slider('select town of 01',options=[0,1], key='embarked_slider_Q')
embarked_S = st.sidebar.select_slider('select town of 01',options=[0,1], key='embarked_slider_S')
embark_town_Queenstown = st.sidebar.select_slider('select town of 01',options=[0,1], key='town_slider_Q')
embark_town_Southampton = st.sidebar.select_slider('select town of 01',options=[0,1], key='town_slider_S')

input = np.array([pclass,age,sibsp,sex_male,fare,class_label,parch,embarked_Q,embarked_S,embark_town_Queenstown,embark_town_Southampton]).reshape(1,-1)

if st.button('Predict'):
    result = Titanic_model.predict(input)[0]
    prediction = {1 :"Survived",0:"Not Survived"}

    st.write(prediction[result])

#prediction = Titanic_model.predict(features)
#if prediction == 1:
    #st.success('Survived(1)')
#else:
    #st.error('Did not survive(0)')

