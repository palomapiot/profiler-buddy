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
    """
    Get demographic data from profile (automatic profiling)
    ---
    get:
        summary: Get the demographic from user feed
        description: Predicts demographic data based on the user feed
        tags:
          - Profile
        operationId: profile
        parameters:[]
        requestBody:
            content:
            application/json:
                schema:
                $ref: '#/components/schemas/Comments'
        responses:
            '200':
                description: Demographic profile
                schema: 
                    $ref: '#/components/schemas/DemographicData'
    """
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
    """
    Beck depression inventory automatic filling
    ---
    get:
        summary: Fill Beck depression inventory from user feed
        description: Predicts the answers of Beck's depression inventory
        tags:
          - Questionnaire
        operationId: questionnaire
        parameters: []
        requestBody:
            content:
            application/json:
                schema:
                $ref: '#/components/schemas/Comments'
        responses:
            '200':
                description: Beck depression inventory answers and contexts
                schema: 
                    $ref: '#/components/schemas/Questionnaire'
    """
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