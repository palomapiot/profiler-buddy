import flask
import os
import pickle
import lightgbm

from flask import request, jsonify
from classifiers.gender.gender import preprocess

GENDER_MODEL = pickle.load(open(os.path.join('classifiers/gender', 'gender77'), 'rb'))

app = flask.Flask(__name__)
app.config["DEBUG"] = True

comments = {
    "comments": [ 
        { 
        "comment": "adsfsfdasdfasdfadsf",
        "date": "01-01-2020"
        },
        { 
        "comment": "hola hola hola no vengas sola",
        "date": "01-01-2020"
        },
        { 
        "comment": "let's gooooooooooooooooo",
        "date": "01-01-2020"
        }
    ],
    "experiment_id": "test"
}

@app.route('/profile', methods=['POST'])
def home():
    data = request.get_json() or {}
    comment_list = [x['comment'] for x in data['comments']]
    df = preprocess(data['experiment_id'], ' '.join(comment_list))
    print(df)
    # predict
    gender = GENDER_MODEL.predict(df.drop(['text', 'author_id'], axis=1))
    return Response({'result': 'OK', 'gender': gender}, status=HTTP_200_OK)
    return jsonify(data['comments'])

app.run(host='0.0.0.0')