import json

import numpy as np
from flask import Flask, request, send_from_directory
from flask_cors import CORS

from NeuralNetwork.Architectures.LSTM.LSTM import LSTM
from NeuralNetwork.Architectures.GRU.GRU import GRU
from NeuralNetwork.NeuralNetwork import NeuralNetwork

app = Flask(__name__)
CORS(app)


@app.route('/')
def index():
    return 'This is the Psychobot API.\nsentiment - sentiment analysis'


@app.route('/<path:path>')
def send_validation(path):
    return send_from_directory("validate", path)


@app.route('/sentiment')
def sentiment():
    sentence = request.args.get('sentence', '')

    emotion, feelings = ml.run_model_with_embedding(sentence)

    return {
        'res': {
            'sentence': sentence,
            'feeling': emotion,
            'reliability': feelings
        }
    }


@app.route('/user/register')
def register():
    username = request.args.get('username', '')
    password = request.args.get('password', '')
    email = request.args.get('email', '')

    return {
        "res": {
            "type": "success"
        }
    }


@app.route('/user/login')
def login():
    email = request.args.get('email', '')
    password = request.args.get('password', '')

    return {
        "res": {
            "type": "success"
        }
    }


@app.route('/user/update')
def update():
    config = request.args.get('config', {})

    return {
        "res": {
            "type": "success"
        }
    }


def separate_dataset_to_batches(dataset, batch_size):
    batches = []
    for i in range(0, len(dataset), batch_size):
        batches.append(dataset[i:i + batch_size])
    return batches


if __name__ == '__main__':
    global ml
    # initialize ML

    list_of_feelings = []
    with open(r'./30k-happy-sadness-anger/list.json', 'r') as f:
        list_of_feelings = json.load(f)

    ml = NeuralNetwork(GRU(list_of_feelings, learning_rate=1, embed=True))
    examples = np.load(r'./30k-happy-sadness-anger/data.npy', allow_pickle=True)

    examples = separate_dataset_to_batches(examples[0], 100)

    ml.train(examples[:int(len(examples) * 0.8)], 15)

    app.run(host='0.0.0.0', port=8080)
    # serve(app, host='0.0.0.0', port=80, ssl_context=('cert.crt', 'cert.key'))
