from flask import Flask, request, render_template
import keras
from keras.models import load_model
import numpy as np
global model, graph, sess
import tensorflow as tf
import re, nltk
import pickle
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
nltk.download('stopwords')

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

app = Flask(__name__)

@app.route('/', methods =['POST', 'GET'])
def predict():
    if request.method=="POST":
        tweet = request.form["tweet"]
        print(tweet)
        model = tf.keras.models.load_model('Embedding_model_final.h5')
        with open("tokenizer.pickle", 'rb') as handle:
            tokenizer = pickle.load(handle)
        review= re.sub('[^a-zA-Z]', ' ', tweet)
        review= review.lower()
        review= review.split()
        review=[word for word in review if not word in set(stopwords.words('english'))]
        ps= PorterStemmer()
        review =[ps.stem(word) for word in review if not word in set(stopwords.words('english'))]
        review=' '.join(review)

        seq = tokenizer.texts_to_sequences([review])
        padded_data = pad_sequences(seq, padding="post", maxlen = 40)
        y= model.predict(padded_data)
        return render_template('home.html', y=y)
    else:
        y=0

    return render_template('home.html', y=y)


if __name__ == '__main__':
    app.run(debug = True)