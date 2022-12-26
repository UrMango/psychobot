from flask import Flask, request, send_from_directory
from flask_cors import CORS

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
    return {
        'res': {
            'sentence': sentence,
            'feeling': 'natural',
            'reliability': {
                'anger': 0.0,
                'disgust': 0.0,
                'fear': 0.0,
                'joy': 0.0,
                'sadness': 0.0,
            }
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


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080)
    # serve(app, host='0.0.0.0', port=80, ssl_context=('cert.crt', 'cert.key'))
