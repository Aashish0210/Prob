import streamlit as st
import yfinance as yf
import pandas as pd
from sklearn import datasets
from sklearn.ensemble import RandomForestClassifier

# st.write("""
# ## Stock of Google using streamlit and yfinance
#          Here is the information about close and volume of the stock
# """ )
# tickerSource='GOOGL'
# data=yf.Ticker(tickerSource)
# tickerFrame=data.history(period='1d',start='2010-1-30', end='2020-1-20')
# st.line_chart(tickerFrame['Close'])
# st.line_chart(tickerFrame['Volume'])

# Title
st.write("""
# Simple Wine Quality Prediction App
This app predicts the **Wine Quality** type!
""")

# Sidebar for user input
st.sidebar.header('User Input for Prediction')

def user_input():
    # Capture all 13 features from the Wine dataset
    alcohol = st.sidebar.slider('Alcohol', 11.0, 15.0, 13.0)
    malic_acid = st.sidebar.slider('Malic Acid', 0.74, 5.80, 2.0)
    ash = st.sidebar.slider('Ash', 1.36, 3.23, 2.36)
    alcalinity_of_ash = st.sidebar.slider('Alcalinity of Ash', 10.6, 30.0, 19.5)
    magnesium = st.sidebar.slider('Magnesium', 70.0, 162.0, 99.0)
    total_phenols = st.sidebar.slider('Total Phenols', 0.98, 3.88, 2.3)
    flavanoids = st.sidebar.slider('Flavanoids', 0.34, 5.08, 2.0)
    nonflavanoid_phenols = st.sidebar.slider('Nonflavanoid Phenols', 0.13, 0.66, 0.4)
    proanthocyanins = st.sidebar.slider('Proanthocyanins', 0.41, 3.58, 1.65)
    color_intensity = st.sidebar.slider('Color Intensity', 1.28, 13.0, 5.0)
    hue = st.sidebar.slider('Hue', 0.48, 1.71, 1.0)
    od280_od315_of_diluted_wines = st.sidebar.slider('OD280/OD315 of diluted wines', 1.27, 4.0, 2.78)
    proline = st.sidebar.slider('Proline', 278.0, 1680.0, 750.0)
    
    # Store features in a dictionary
    data = {
        'Alcohol': alcohol,
        'Malic Acid': malic_acid,
        'Ash': ash,
        'Alcalinity of Ash': alcalinity_of_ash,
        'Magnesium': magnesium,
        'Total Phenols': total_phenols,
        'Flavanoids': flavanoids,
        'Nonflavanoid Phenols': nonflavanoid_phenols,
        'Proanthocyanins': proanthocyanins,
        'Color Intensity': color_intensity,
        'Hue': hue,
        'OD280/OD315 of diluted wines': od280_od315_of_diluted_wines,
        'Proline': proline
    }

    # Convert to DataFrame
    features = pd.DataFrame(data, index=[0])
    return features

# Display user input
df = user_input()
st.subheader('User Input')
st.write(df)

# Load the wine dataset
wine = datasets.load_wine()
x = wine.data
y = wine.target

# Display wine classes
st.write('The list of the wine grades are:')
st.write(wine.target_names)

# Train a Random Forest Classifier
cls = RandomForestClassifier()
cls.fit(x, y)

# Make predictions and show probability
predict = cls.predict(df)
probability = cls.predict_proba(df)

# Display prediction and probability
st.subheader('Prediction')
st.write(wine.target_names[predict])

st.subheader('Prediction Probability')
st.write(probability)



st.write("""
# Simple Iris flower Predicton App
         This app predicts the **Iris flower** type!
""")
st.sidebar.header('User input Prediction')
def user_input():
    sepal_length=st.sidebar.slider('sepal_length',4.3,7.9,5.4)
    sepal_width=st.sidebar.slider('speal_width',2.0,4.4,3.4)
    petal_length=st.sidebar.slider('petal_length',1.0,6.9,1.3)
    petal_width=st.sidebar.slider('petal_width',1.0,6.9,1.3)

    data={
        'Class0':sepal_length,
        'Class1':sepal_width,
        'Class2':petal_length,
        'class3':petal_width,
    }
    features=pd.DataFrame(data,index=[0])
    return features

df = user_input()
st.subheader('User Input')
st.write(df)
iris=datasets.load_iris()
X=iris.data
Y=iris.target

clf=RandomForestClassifier()
clf.fit(X,Y)

prediction=clf.predict(df)
prediction_probabiiity=clf.predict_proba(df)

st.subheader('Class label and their corresponding index number')
st.write(iris.target_names)

st.subheader('Prediction')
st.write(iris.target_names[prediction])

st.subheader('Prediction Probability')
st.write(prediction_probabiiity)