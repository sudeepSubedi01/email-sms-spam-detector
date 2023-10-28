import streamlit as st
import pickle
import nltk
from nltk.stem.porter import PorterStemmer
import string
from nltk.corpus import stopwords


tfidf = pickle.load(open('tfidf_vectorizer.pkl','rb'))
mnb = pickle.load(open('multinomial_nb.pkl','rb'))


def transform_text(text):
  text = text.lower()
  text = nltk.word_tokenize(text)

  y = []
  for i in text:
    if i.isalnum():
      y.append(i)
  text = y[:]

  y.clear()
  for i in text:
    if i not in stopwords.words('english') and i not in string.punctuation:
      y.append(i)

  text = y[:]

  y.clear()
  ps = PorterStemmer()
  for i in text:
    y.append(ps.stem(i))
      
  text = y[:]

  text = " ".join(text)
  return text

st.title('Email/SMS Spam Detector')

user_text = st.text_area('Enter your email/sms here:')
user_text = transform_text(user_text)

X = tfidf.transform([user_text]).toarray()
y_pred = mnb.predict(X)[0]

if st.button('Predict'):
  if y_pred == 0:
    st.write('Not Spam')
  else:
    st.write('Spam')
