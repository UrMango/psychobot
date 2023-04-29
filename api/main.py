import json
import pickle
from enum import Enum

from flask import Flask, request, send_from_directory
from flask_cors import CORS

import wandb

import asyncio

from NeuralNetwork.Architectures.GRU.GRU import GRU
from NeuralNetwork.Architectures.NEW_LSTM.NEW_LSTM import NEW_LSTM
from Dataset.dataset_loader import Dataset

EPOCHS = 15

NUMBER_OF_EXAMPLES_IN_BATCH = 100

BATCH_SIZE = 60

EXAMPLES = 20000

TRAINING_SET_PERCENTAGE = 0.8

IsExistResults = {"exist": 1, "not_exist": 2, "training": 3}
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
    architecture = request.args.get('arch', 'recommended')
    feelings = json.loads(request.args.get('feelings', '["happy", "sadness", "anger"]'))
    feelings.sort()  # sort the feelings by alphabetical order
    sentence = request.args.get('sentence', '')



    emotion = None
    result_feelings = None
    if architecture == "recommended":
        emotion, result_feelings = recommendedMl.run_model_with_embedding(sentence)
    else:
        dict_parameters = download_model(architecture, feelings)
        speceficMl = None
        if architecture == "LSTM":
            speceficMl = NEW_LSTM(feelings, set_parameters=True, parameters=dict_parameters, embed=True, _nlp=nlp, _model=model_embedding)
        elif architecture == "GRU":
            speceficMl = GRU(feelings, set_parameters=True, parameters=dict_parameters, embed=True, _nlp=nlp, _model=model_embedding)
        emotion, result_feelings = speceficMl.run_model_with_embedding(sentence)
    return {
        'res': {
            'sentence': sentence,
            'feeling': emotion,
            'reliability': result_feelings
        }
    }

@app.route('/is_exist')
def is_exist():
    architecture = request.args.get('arch', '')
    feelings = json.loads(request.args.get('feelings', '["happy", "sadness", "anger"]'))
    feelings.sort()  # sort the feelings by alphabetical order
    is_model_exist = IsExistResults["not_exist"]
    percent = 0
    metadata = None
    print('noamr/model-registry/' + architecture.lower() + "-" + str(feelings).replace(", ", "") + ':latest')
    try:
        artifact = wandb.use_artifact('noamr/model-registry/' + architecture.lower() + '-' + str(feelings).replace(", ", "") + ':latest',
                           type='model')
        metadata = artifact.metadata
        is_model_exist = IsExistResults["exist"]
    except wandb.errors.CommError:
        try:
            artifact = wandb.use_artifact(
                'noamr/psychobot/' + architecture.lower() + '-' + str(feelings).replace(", ", "") + ':latest',
                type='model')
            metadata = artifact.metadata
            is_model_exist = IsExistResults["exist"]
        except wandb.errors.CommError:
            if architecture == "GRU":
                for model in GRU.arrayInstances:
                    if sorted(model.list_of_feelings) == sorted(feelings):
                        is_model_exist = IsExistResults["training"]
                        percent = model.percent
                        break
            elif architecture == "LSTM":
                for model in NEW_LSTM.arrayInstances:
                    if sorted(model.list_of_feelings) == sorted(feelings):
                        is_model_exist = IsExistResults["training"]
                        percent = model.percent
                        break

    return {
        'res': {
            'is_exist': is_model_exist,
            'percent': percent,
            'metadata': metadata
        }
    }


@app.route('/train-custom')
def train_custom():
    architecture = request.args.get('arch', '')
    feelings = json.loads(request.args.get('feelings', '["happy", "sadness", "anger"]'))
    feelings.sort()  # sort the feelings by alphabetical order
    num_of_examples = int(request.args.get('num_of_examples', str(EXAMPLES)))

    try:
        wandb.use_artifact('noamr/model-registry/' + architecture.lower() + '-' + str(feelings).replace(", ", "") + ':latest', type='model')
        return {
            'res': {
                'text': "Already trained!"
            }
        }
    except wandb.errors.CommError:
        try:
            artifact = wandb.use_artifact(
                'noamr/psychobot/' + architecture.lower() + '-' + str(feelings).replace(", ", "") + ':latest',
                type='model')
            return {
                'res': {
                    'text': "Already trained!"
                }
            }
        except wandb.errors.CommError:
            asyncio.run(run_build_train_model(num_of_examples, feelings, architecture))
            return {
                'res': {
                    'text': "Started training..."
                }
            }

async def run_build_train_model(num_of_examples, feelings, architecture):
    asyncio.create_task(build_train_model(num_of_examples, feelings, architecture))

async def build_train_model(num_of_examples, feelings, architecture):
    print("Entered build_train_model")
    model = None
    if architecture.lower() == "gru":
        print("Entered GRU")
        model = GRU(feelings, embed=True, _nlp=nlp, _model=model_embedding)
        print("Finished build GRU model")
    elif architecture.lower() == "lstm":
        model = NEW_LSTM(feelings, embed=True, _nlp=nlp, _model=model_embedding)
    examples = Dataset.make_examples(model, num_of_examples, feelings)
    print(examples[0])
    batches = []
    for i in range(0, len(examples), BATCH_SIZE):
        batches.append(examples[i:i + BATCH_SIZE])
    examples = separate_dataset_to_batches(examples, BATCH_SIZE)
    print(examples[0])
    dataset_name = str(num_of_examples % 1000) + "k-" + architecture + "-" + str(feelings)
    model.train(examples[:int(len(examples) * TRAINING_SET_PERCENTAGE)], examples[int(len(examples) * TRAINING_SET_PERCENTAGE):], BATCH_SIZE, EPOCHS, dataset_name)




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


def download_model(arch='gru', feelings=None):
    if feelings is None:
        feelings = ['happy', 'sadness', 'anger']
    feelings.sort()
    feelings = str(feelings).replace(", ", "")
    try:
        artifact = wandb.use_artifact('noamr/model-registry/' + arch.lower() + "-" + feelings + ':latest', type='model')
    except wandb.errors.CommError:
        artifact = wandb.use_artifact('noamr/psychobot/' + arch.lower() + "-" + feelings + ':latest', type='model')
    artifact_dir = artifact.download("models/latest")

    got_dict_parameters = None
    got_list_of_feelings = []

    with open(artifact_dir + '/dataset/list.json', 'r') as f:
        got_list_of_feelings = json.load(f)

    print(artifact.metadata)
    
    with open(artifact_dir + "/model/parameters_" + str(got_list_of_feelings) + ".json", 'rb') as f:
        got_dict_parameters = pickle.load(f)

    return got_dict_parameters
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
    # initialize ML
    wandb.init(job_type="api-run")

    recommend_list_of_feelings = ['happy', 'sadness', 'anger']

    recommend_dict_parameters = download_model()

    #lstm_dict_parameters, lstm_list_of_feelings = download_model("lstm")

    recommendedMl = GRU(recommend_list_of_feelings, set_parameters=True, parameters=recommend_dict_parameters, embed=True)
    nlp = recommendedMl.nlp
    model_embedding = recommendedMl.model
    #mlLSTM = NEW_LSTM(lstm_list_of_feelings, set_parameters=True, parameters=lstm_dict_parameters, embed=True)

    app.run(host='0.0.0.0', port=8080)
    # serve(app, host='0.0.0.0', port=80, ssl_context=('cert.crt', 'cert.key'))
