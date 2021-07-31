from flask import Flask
from flask_restful import Api, Resource, reqparse

app = Flask(__name__)
api = Api(app)

returnParse = reqparse.RequestParser()
returnParse.add_argument("action", type=int, required=True)

returnVal = {}

class getMethod(Resource):
    def get(self, id):
        return returnVal[id]

    def put(self, id):
        arg = returnParse.parse_args()
        returnVal[id] = arg.action
        return returnVal[id]

returnVal[1] = 0
api.add_resource(getMethod, "/<int:id>")

if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0', port=56789)
