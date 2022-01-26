from flask import Flask
from flask_restful import Api
from resources.register import FaceRegister
from resources.recognition import FaceRecognition
from flask_cors import CORS
app = Flask(__name__)
CORS(app)
api = Api(app)
api.add_resource(FaceRecognition, '/image')
api.add_resource(FaceRegister, '/upload')
if __name__ == "__main__":
    app.run(port='3001', debug=False)
