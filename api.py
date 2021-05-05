import random
import json

import torch

from chatbot_util.model import NeuralNetwork
from chatbot_util.nltk_wrapper import bag_of_words, tokenise
from flask import Flask
from flask import jsonify
from flask_ngrok import run_with_ngrok
import requests

app = Flask(__name__)
run_with_ngrok(app)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

with open('chatbot_util/train.json', 'r') as json_data:
    intents = json.load(json_data)

FILE = "chatbot_util/data.pth"
data = torch.load(FILE)

input_size = data["input_size"]
hidden_size = data["hidden_size"]
output_size = data["output_size"]
all_words = data['all_words']
tags = data['tags']
model_state = data["model_state"]

model = NeuralNetwork(input_size, hidden_size, output_size).to(device)
model.load_state_dict(model_state)
model.eval()


@app.route('/')
def greet():
    return 'Hoi!'


@app.route('/chatbot/<input_message>')
def index(input_message):
    input_message = tokenise(input_message)

    X = bag_of_words(input_message, all_words)
    X = X.reshape(1, X.shape[0])
    X = torch.from_numpy(X).to(device)

    output = model(X)
    _, predicted = torch.max(output, dim=1)

    tag = tags[predicted.item()]

    probs = torch.softmax(output, dim=1)
    prob = probs[0][predicted.item()]

    output_message = ""

    if prob.item() > 0.75:
        for intent in intents['classes']:
            if tag == intent["tag"]:
                output_message = random.choice(intent['responses'])
    else:
        output_message = "Sorry, write in simple words pls"

    json_output = dict()
    json_output['reply'] = output_message

    if output_message == "MEME":
        nsfw = True
        is_gif = True
        url = ""
        while nsfw or is_gif:
            response = requests.get("https://meme-api.herokuapp.com/gimme")
            print(response.json())
            nsfw = response.json()['nsfw']
            url = response.json()['url']

            if url.split('.')[-1] != 'gif':
                is_gif = False

            json_output['reply'] = url

    return jsonify(json_output)


if __name__ == "__main__":
    app.run()
