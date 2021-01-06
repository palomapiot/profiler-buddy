import flask
import os
import pickle
import lightgbm

from flask import request, jsonify
from preprocessing.preprocessing import preprocess
from classifiers.depression.depression import find_answers

GENDER_MODEL = pickle.load(open(os.path.join('classifiers/gender', 'gender77'), 'rb'))

app = flask.Flask(__name__)
app.config["DEBUG"] = True

@app.route('/profile', methods=['POST'])
def profile():
    data = request.get_json() or {}
    comments = [x['comment'] for x in data['comments']]
    df = preprocess(data['experiment_id'], ' '.join(comments))
    
    # predict
    gender = GENDER_MODEL.predict(df.drop(['text', 'author_id'], axis=1))
    return Response({'result': 'OK', 'gender': gender}, status=HTTP_200_OK)

@app.route('/questionnaire', methods=['POST'])
def fill_questionnaire():
    data = request.get_json() or {}
    comments = [x['comment'] for x in data['comments']]
    questionnaire = find_answers(' '.join(comments))
    return Response({'result': 'OK', 'questionnaire': questionnaire}, status=HTTP_200_OK)

@app.route('/', methods=['GET'])
def home():
    return "<h1>Profiler Buddy</h1>"

app.run(host='0.0.0.0')