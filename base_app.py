"""

    Simple Streamlit webserver application for serving developed classification
	models.

    Author: Explore Data Science Academy.

    Note:
    ---------------------------------------------------------------------
    Please follow the instructions provided within the README.md file
    located within this directory for guidance on how to use this script
    correctly.
    ---------------------------------------------------------------------

    Description: This file is used to launch a minimal streamlit web
	application. You are expected to extend the functionality of this script
	as part of your predict project.

	For further help with the Streamlit framework, see:

	https://docs.streamlit.io/en/latest/

"""
# Streamlit dependencies
import streamlit as st
import joblib,os
from PIL import Image
import json
import requests
from streamlit_lottie import st_lottie

# Data dependencies
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import re
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from nltk.tokenize import word_tokenize
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.stem import PorterStemmer
from nltk import pos_tag

#Data Visualization
import seaborn as sns
from scipy import stats
import matplotlib.pyplot as plt
from sklearn.metrics import plot_roc_curve
from wordcloud import WordCloud
import warnings
warnings.filterwarnings('ignore')

def load_lottieurl(url: str):
    r = requests.get(url)
    if r.status_code != 200:
        return None
    return r.json()

lottie_hello = load_lottieurl("https://assets9.lottiefiles.com/packages/lf20_M9p23l.json")

st_lottie(
    lottie_hello,
    speed=1,
    reverse=False,
    loop=True,
    quality="low", # medium ; high
    height=None,
    width=None,
    key=None,
)

# Vectorizer
news_vectorizer = open("resources/tfidfvect.pkl","rb")
tweet_cv = joblib.load(news_vectorizer) # loading your vectorizer from the pkl file

# Load your raw data
raw = pd.read_csv("resources/train.csv")

# The main function where we will build the actual app
def main():
	"""Tweet Classifier App with Streamlit """

	# Creates a main title and subheader on your page -
	# these are static across all pages
	st.title("Tweet Classifier")
	st.subheader("Climate change tweet classification")

	# Creating sidebar with selection box -
	# you can create multiple pages this way
	options = ["Prediction", "Information", "Data Visualisation", "Enquiries"]
	selection = st.sidebar.selectbox("Dashboard", options)
	st.sidebar.image("logo.jpg", use_column_width=True)

	# Building out the "Information" page
	if selection == "Information":

		st.info("General Information")
		# You can read a markdown file from supporting resources folder
		st.markdown("Some information here")

		st.subheader("Classification")
		# You can read a markdown file from supporting resources folder
		st.markdown("Classification is defined as the process of recognition, understanding, and grouping of objects and ideas into preset categories.")
		st.write("Follow this [link](https://pythonguides.com/scikit-learn-classification/) for more info.")
		st.image("Data_Classification.jpg", use_column_width=True)

		st.subheader('Logistic Regression Classifier')
		st.write("Follow this [link](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html?highlight=logistic%20regression#sklearn.linear_model.LogisticRegression) for more info on Logistic Regression Classifier.")
		
		st.subheader('Support Vector Classifier')
		st.write("Follow this [link](https://towardsdatascience.com/svm-support-vector-machine-for-classification-710a009f6873) for more info on K-Nearest Neighbors Classifier.")
		
		st.subheader('Other Classifiers')
		st.write("Follow this [link](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html) for more info on Random Forest Classifier.")
		
		st.write("Follow this [link](https://towardsdatascience.com/naive-bayes-explained-9d2b96f4a9c0) for more info on Naïve Bayes Classifier.")
	
		st.write("Follow this [link](https://scikit-learn.org/stable/modules/generated/sklearn.impute.KNNImputer.html?highlight=knn#sklearn.impute.KNNImputer) for more info on K-Nearest Neighbors Classifier.")
		

		st.subheader("Raw Twitter data and label")
		if st.checkbox('Show raw data'): # data is hidden if box is unchecked
			st.write(raw[['sentiment', 'message']]) # will write the df to the page

	# Building out the predication page
	if selection == "Prediction":
		st.subheader("Prediction with ML Models")
		
		st.markdown('Instructions:')
		st.markdown('1. Check the box below to display the raw data')
		st.markdown('2. Copy a tweet by clicking the checkbox below and copying tweet into textbox.')
		st.markdown('3. Choose a prediction model to use.')
		st.markdown('4. Learn if the tweet supports or opposes man-made climate change.')


		
		st.subheader('Check the box below!')
		if st.checkbox('Show raw data'): # checke the box to display data
			st.write(raw[['message', 'sentiment']]) # will write the df to the page

		# Creating a text box for user input
		tweet_text = st.text_area("Enter/paste text below:")
		st.markdown('NB: Please use Ctrl + enter, to save your predictions.')
		
		if st.button('Logistic Regression Classifer'):
			# Transforming user input with vectorizer
			vect_text = tweet_cv.transform([tweet_text]).toarray()
			# Load your .pkl file with the model of your choice + make predictions
			# Try loading in multiple models to give the user a choice
			predictor = joblib.load(open(os.path.join("resources/Logistic_regression.pkl"),"rb"))
			prediction = predictor.predict(vect_text)
			if prediction == -1:
				result = 'Anti: the tweet does not believe in man-made climate change Variable definitions'	
			elif prediction == 0:
				result = 'Neutral: the tweet neither supports nor refutes the belief of man-made climate change'
			elif prediction == 1:
				result = 'Pro: the tweet supports the belief of man-made climate change'
			else:
				result = ' News: the tweet links to factual news about climate change'	
				
			st.success(result)

		if st.button("Support Vector Classifier"):
			# Transforming user input with vectorizer
			
			predictor = joblib.load(open(os.path.join("resources/finalized_model"),"rb"))
			prediction = predictor.predict([tweet_text])
			if prediction == -1:
				result = 'Anti: the tweet does not believe in man-made climate change Variable definitions'	
			elif prediction == 0:
				result = 'Neutral: the tweet neither supports nor refutes the belief of man-made climate change'
			elif prediction == 1:
				result = 'Pro: the tweet supports the belief of man-made climate change'
			else:
				result = ' News: the tweet links to factual news about climate change'	
				
			st.success(result)

		


	if selection == "Data Visualisation":
		st.info("Visual Analysis")
		st.markdown('In this section we take a closer look at our data using visual analysis ')

		
		raw['sentiment'] = [['Negative', 'Neutral', 'Positive', 'News'][x+1] for x in raw['sentiment']]# checking the distribution
		st.subheader('Pie chart visual of sentiment distribution')
		st.set_option('deprecation.showPyplotGlobalUse', False)
		values = raw['sentiment'].value_counts()/raw.shape[0]
		labels = (raw['sentiment'].value_counts()/raw.shape[0]).index
		colors = ['red', 'maroon', 'brown', 'orange']
		plt.pie(x=values, labels=labels, autopct='%1.1f%%', startangle=90, explode= (0.1, 0.1, 0.1, 0.1), colors=colors)
		st.pyplot()

		st.markdown('Positive(Pro) - 52.3% of the tweets support the belief of man-made climate change ')
		st.markdown('News - 21.1% of the tweets link to factual news about climate change')
		st.markdown('Neutral - 17.6% of the tweets neither support nor refute the belief of man-made climate change')
		st.markdown('Negative(Anti) - 9.1% of the tweets does not believe in man-made climate change Variable definitions.')
		

		st.subheader('Wordcloud visual of sentiment distribution')
		st.markdown('NB: The word cloud is a visualisation that represents the most frequently occuring words in the given dataset')
		st.markdown('')
		all_words = " ".join([sentence for sentence in raw["message"]])
		wordcloud = WordCloud(width=800, height=500, random_state=42, max_font_size=100).generate(all_words)
		#plot graph
		plt.figure(figsize=(15,8))
		plt.imshow(wordcloud, interpolation= 'bilinear')
		plt.axis('off')
		st.pyplot()
		
		st.subheader('Top Three Words:')
		st.markdown('Climate change')
		st.markdown('Global warming')
		st.markdown('Believe')

		st.subheader("Raw Twitter data and label")
		st.write(raw[['sentiment', 'message']])

		st.image("th.jpg", use_column_width=True)



	if selection == 'Enquiries':
		
		st.write('For any questions or queries, please contact any of our staff members and we will be in touch shortly:')
		
		st.subheader('Web Designer')
		st.write('Theodore Ramahanedza')
		st.write('Email: tkhuliso753@gmail.com')
		
		st.subheader('AI Specialist')
		st.write('Lebogang Methikge')
		st.write('Email: lebomethikge@gmail.com')
		
		st.subheader('Data Scientist')
		st.write('Philile Luhlanga')
		st.write('Email: ppluhlanga@gmail.com')
		
		st.subheader('Data Scientist')
		st.write('Obafemi Deborah')
		st.write('Email: obafemideborah47@gmail.com')
		
		st.subheader('Data Analyst')
		st.write('Robert Ochieng')
		st.write('Email: robochi6@gmail.com')
		
		st.subheader('Data Engineer')
		st.write('Elizabeth Olorunleke')
		st.write('Email: Elizabeth.olorunleke@hotmail.com')

		

# Required to let Streamlit instantiate our web app.  
if __name__ == '__main__':
	main()
