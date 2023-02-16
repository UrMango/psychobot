import json
import pickle

from flask import Flask, request, send_from_directory
from flask_cors import CORS

import wandb

from NeuralNetwork.Architectures.GRU.GRU import GRU

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


def download_model():
    artifact = wandb.use_artifact('noamr/model-registry/psychobot:latest', type='model')
    artifact_dir = artifact.download("models/latest")

    got_dict_parameters = None
    got_list_of_feelings = []
    with open(artifact_dir + '/dataset/list.json', 'r') as f:
        got_list_of_feelings = json.load(f)
    with open(artifact_dir + "/model/parameters_" + str(got_list_of_feelings) + ".json", 'rb') as f:
        got_dict_parameters = pickle.load(f)

    return got_dict_parameters, got_list_of_feelings
#
#
# def get_model():
#     artifact = wandb.use_artifact('noamr/model-registry/psychobot:latest', type='model')
#
#     got_list_of_feelings = json.load(artifact.get('dataset/list.json'))
#     got_dict_parameters = pickle.load(artifact.get("model/parameters_" + str(list_of_feelings) + ".json"))
#
#     return got_dict_parameters, got_list_of_feelings


if __name__ == '__main__':
    global ml
    # initialize ML

    wandb.init(job_type="api-run")

    dict_parameters, list_of_feelings = download_model()

    ml = GRU(list_of_feelings, set_parameters=True, parameters=dict_parameters, embed=True)

    app.run(host='0.0.0.0', port=8080)
    # serve(app, host='0.0.0.0', port=80, ssl_context=('cert.crt', 'cert.key'))
