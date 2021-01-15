import flask
import os
import pickle
import lightgbm
import numpy
from status import HTTP_200_OK

from flask import request, jsonify, Response
from preprocessing.preprocessing import preprocess
from classifiers.depression.depression import find_answers

# male = 0, female = 1
GENDER_MODEL = pickle.load(open(os.path.join('classifiers/gender', 'gender77'), 'rb'))

app = flask.Flask(__name__)
app.config["DEBUG"] = True

@app.route('/profile', methods=['POST'])
def profile():
    print('profile call')
    data = request.get_json() or {}
    comments = [x['text'] for x in data['comments']]
    df = preprocess(data['experiment_id'], ' '.join(comments))
    
    # predict
    gender_score = GENDER_MODEL.predict(df.drop(['text', 'author_id'], axis=1))
    if round(gender_score[0], 0) == 1.0:
        gender = 'Male'
    else:
        gender = 'Female'
    return jsonify( {'gender': gender, 'score': str(gender_score[0]) } )

@app.route('/questionnaire', methods=['POST'])
def fill_questionnaire():
    print('fill questionnaire call')
    data = request.get_json() or {}
    comments = [x['text'] for x in data['comments']]
    questionnaire = find_answers(comments)
    data = jsonify(questionnaire)
    return data

@app.route('/', methods=['GET'])
def home():
    return "<h1>Profiler Buddy</h1>"

app.run(host='0.0.0.0')