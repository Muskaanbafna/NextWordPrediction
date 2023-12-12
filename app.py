from flask import Flask
from flask import render_template
import pickle
from keras.models import load_model
import numpy as np
from tensorflow.keras.preprocessing.sequence import pad_sequences
from flask import Flask, render_template, request, jsonify

app = Flask(__name__)

model = load_model('assets/word_model.h5')

with open('assets/tokenizer.pickle', 'rb') as handle:
    tokenizer = pickle.load(handle)

with open('assets/max_sequence_len.pickle', 'rb') as handle:
    max_sequence_len = pickle.load(handle)


@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict_word', methods=['POST'])
def predict_word():
    data = request.get_json()
    word = data['word']

    token_list = tokenizer.texts_to_sequences([word])[0]
    token_list = pad_sequences([token_list], maxlen=max_sequence_len-1, padding='pre')

    predicted_probabilities = model.predict(token_list, verbose=0)
    predicted_index = np.argmax(predicted_probabilities)

    output_word = ""
    for word, index in tokenizer.word_index.items():
        if index == predicted_index:
            output_word = word
            break

    return jsonify({'predictedWord': output_word})

def predict_word(word):
    for _ in range(word):
        token_list = tokenizer.texts_to_sequences([word])[0]
        token_list = pad_sequences([token_list], maxlen=max_sequence_len-1, padding='pre')
        
        # Use model.predict instead of model.predict_classes
        predicted_probabilities = model.predict(token_list, verbose=0)
        
        # Find the index of the word with the highest probability
        predicted_index = np.argmax(predicted_probabilities)
        
        output_word = ""
        for word, index in tokenizer.word_index.items():
            if index == predicted_index:
                output_word = word
                break

    return(output_word)