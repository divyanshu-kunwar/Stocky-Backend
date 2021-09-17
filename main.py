from flask import Flask, json , jsonify , request
import graphlib.financialGraph as fg
import pandas as pd
import indicatorlib.indicators as ind

app = Flask(__name__)

@app.route('/graph/<string:graphtype>', methods=['POST'])
def sendGraphData(graphtype):

    receievedData = request.json
    df = pd.read_json(receievedData,orient = 'split')
    graphObj = fg.Data(df , graphtype)

    return jsonify(graphObj.send_data().to_json())

@app.route('/indicator/<string:indicator_name>',
 methods=['POST'])
def sendIndData(indicator_name):
    try:
        period = request.args.get('period')
        columns = request.args.get('columns')
        print(indicator_name , period , columns)
    except:
        print("error")
    receievedData = request.json
    df = pd.read_json(receievedData,orient = 'split')
    data_r = calcIndData(df,indicator_name,period,columns)
    return jsonify(data_r.to_json())
def calcIndData(df,indicator_name='atr',period=14,columns="close"):
    if(indicator_name == "atr"):
        return ind.atr(df,period=int(period))
    elif(indicator_name == "apz"):
        return ind.apz(df,period=period)
    elif(indicator_name == "bbands"):
        return ind.bbands(df,period=period,column=columns)
    elif(indicator_name == "dema"):
        return ind.dema(df,period=period,column=columns)
    elif(indicator_name == "dmi"):
        return ind.dmi(df,column=columns)
    elif(indicator_name == "ema"):
        return ind.ema(df,period=period,column=columns)
    elif(indicator_name == "er"):
        ind.er(df,period=period,column=columns)
    elif(indicator_name == "evstc"):
        return ind.evstc(df)
    elif(indicator_name == "evwma"):
        return ind.evwma(df,period=period)

@app.route('/news')
def sendNews():
    return "aaj ki taaza khabar"

if __name__ == "__main__":
    app.run(debug=True)