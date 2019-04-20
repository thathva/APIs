from flask import Flask,jsonify,request
from flask_restful import Api
from flask_restful import Resource,reqparse
from sklearn.externals import joblib

clf2 = joblib.load('model.pkl')
count=joblib.load("counts.pkl")
app = Flask(__name__)
app.config['PROPAGATE_EXCEPTIONS'] = True
api = Api(app)

@app.route('/predict', methods=['POST'])
def predict():
    query=request.get_json()
    a = [query['Message']]
    a=count.transform(a)
    prediction=clf2.predict(a)
    pred="";
    if(str(prediction)==1):
        pred="Spam"
    else:
        pred="Not Spam"
    #return jsonify(query['Message'])
    return jsonify({'prediction':pred})

if __name__ == '__main__':
    
    app.run(port=5000, debug=True)  # important to mention debug=True
