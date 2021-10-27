from flask import Flask, json , jsonify , request
import graphlib.financialGraph as fg
import pandas as pd
import indicatorlib.indicators as ind
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

@app.route('/companyList')
def sendCompanyList():
    companyDF = pd.read_csv("datadownload/equity40.csv")
    return (companyDF.to_json(orient='values'))

@app.route('/initializeData/<string:companyName>', methods=['post'])
def updateTempDatabase(companyName):
    try:
        username = request.args.get('username')
        print(username , " requested for ", companyName)
    except:
        print("error")
    return "Success"

@app.route('/graph/<string:graphType>' , methods=['post'])
def sendGraphData(graphType):
    try:
        timePeriod = request.args.get('period')
        print(graphType , " with timeperiod ", timePeriod)
    except:
        print("error")
    return "Success"

@app.route("/indicator/<string:indicatorName>", methods=['post'])
def sendIndicatorData(indicatorName):
    try:
        timePeriod = request.args.get('timePeriod')
        period = request.args.get('period')
        columns = request.args.get('columns')
        print(indicatorName , period , columns)
    except:
        print("error")
    return "Success"

@app.route("/authentication/userinfo", methods=['post'])
def sendInfo():
    return "Success"

@app.route("/authentication/verify", methods=['post'])
def activateVerification():
    return "Success"

@app.route("/authentication/update", methods=['post'])
def updateInfo():
    return "Success"

@app.route("/news", methods=['post'])
def fetchNews():
    return "Success"

if __name__ == "__main__":
    app.run(debug=True)