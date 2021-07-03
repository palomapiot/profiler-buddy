import flask
import os
import pickle
import lightgbm
import numpy
from status import HTTP_200_OK

from flask import request, jsonify, Response
from preprocessing.preprocessing import preprocess
from classifiers.depression.depression import find_answers, find_answers_for_question

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
    return jsonify( {'gender': gender, 'gender_score': str(gender_score[0]) } )

@app.route('/questionnaire', methods=['POST'])
def fill_questionnaire():
    print('fill questionnaire call')
    data = request.get_json() or {}
    t = [_chunk_string(x['text'], 500) for x in data['comments']]
    comments = [item for sublist in t for item in sublist]
    questionnaire = find_answers(comments)
    data = jsonify(questionnaire)
    return data

@app.route('/train-questionnaire', methods=['POST'])
def train_questionnaire():
    print('train questionnaire call')
    data = request.get_json() or {}
    t = [_chunk_string(x['text'], 500) for x in data['comments']]
    comments = [item for sublist in t for item in sublist]
    questionnaire = find_answers(comments)
    answers = ' '.join(questionnaire['questionnaire'].values())
    result = jsonify(data['experiment_id'] + " " +  answers)
    return result

@app.route('/qa', methods=['POST'])
def predict_answer():
    data = request.get_json() or {}
    texts = [x['text'] for x in data['texts']]
    answer = find_answers_for_question(data['question'], texts)
    data = jsonify(answer)
    return data

@app.route('/', methods=['GET'])
def home():
    return "<h1>Profiler Buddy</h1>"

def _chunk_string(text, length):
    return [text[0 + i:length + i] for i in range(0, len(text), length)]

app.run(host='0.0.0.0')