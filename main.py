from telegram import *
from telegram.ext import *
from sklearn.preprocessing import LabelEncoder
from keras.preprocessing.sequence import pad_sequences
import string
from telegram.ext.updater import Updater
from telegram.ext.messagehandler import MessageHandler
from telegram.ext.filters import Filters

# nlp
import numpy as np
import pandas as pd
import json
import random
from keras.preprocessing.text import Tokenizer
from keras.models import load_model
import numpy as np


updater = Updater("5467511072:AAGHhUbO0WCVUCVFduI7QRtUXfME_kQI9X0",
                  use_context=True)

with open('conversation.json') as content:
    data1 = json.load(content)

tags = []
inputs = []
responses = {}
for intent in data1['intents']:
    responses[intent['tag']] = intent['responses']
    for lines in intent['input']:
        inputs.append(lines)
        tags.append(intent['tag'])

data = pd.DataFrame({"inputs": inputs,
                     "tags": tags})

# removing punctuations
data['inputs'] = data['inputs'].apply(
    lambda wrd: [ltrs.lower() for ltrs in wrd if ltrs not in string.punctuation])
data['inputs'] = data['inputs'].apply(lambda wrd: ''.join(wrd))
# tokenize the data
tokenizer = Tokenizer(num_words=2000)
tokenizer.fit_on_texts(data['inputs'])
train = tokenizer.texts_to_sequences(data['inputs'])

# apply padding
x_train = pad_sequences(train)

# encoding the outputs
le = LabelEncoder()
y_train = le.fit_transform(data['tags'])

# input length
input_shape = x_train.shape[1]


testing = load_model('chatbot.h5')


# func
def ChatBot(update, context):
    texts_p = []
    getinput = update.message.text
    getinput = str(getinput)
    # removing punctuation and converting to lowercase
    prediction_input = [letters.lower(
    ) for letters in getinput
        if letters not in string.punctuation]
    prediction_input = ''.join(prediction_input)
    texts_p.append(prediction_input)

    #tokenizing and padding
    prediction_input = tokenizer.texts_to_sequences(texts_p)
    prediction_input = np.array(prediction_input).reshape(-1)
    prediction_input = pad_sequences([prediction_input], input_shape)

    # getting output from model
    output = testing.predict(prediction_input)
    output = output.argmax()

    # finding the right tag and predicting
    response_tag = le.inverse_transform([output])[0]
    res = random.choice(responses[response_tag])
    res = str(res)
    update.message.reply_text(res)


# test button
updater.dispatcher.add_handler(MessageHandler(Filters.text, ChatBot))
updater.dispatcher.add_handler(MessageHandler(
    Filters.command, ChatBot))  # Filters out unknown commands

updater.start_polling()
