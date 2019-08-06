from flask import Flask,jsonify,request
from flask_restful import Api
from flask_restful import Resource,reqparse
import pandas as pd
from sklearn.externals import joblib
import numpy as np


app = Flask(__name__)
app.config['PROPAGATE_EXCEPTIONS'] = True
api = Api(app)

@app.route('/predict', methods=['POST'])
def predict():
    query=request.get_json()
    a = [query['Message']]
    a=count.transform(a)
    prediction=clf2.predict(a)
    if(prediction==0):
        #pred="Spam"
        return jsonify({'prediction':'Not Spam'})
    #if(str(prediction)=='0'):
    else:
        return jsonify({'prediction':'Spam'})
    #return jsonify(query['Message'])
    #return jsonify({'prediction':str(prediction)})

if __name__ == '__main__':
    clf2 = joblib.load('model.pkl')
    count=joblib.load("counts.pkl")
    app.run(port=5000, debug=True)  # important to mention debug=True
