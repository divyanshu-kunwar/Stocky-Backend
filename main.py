from flask import Flask , jsonify

app = Flask(__name__)

@app.route('/graph/<string:n>')
def calcGraphData(n):
    return jsonify({
        "sent" : n,
        "returned" : n
    })

@app.route('/indicator/<string:n>')
def calcIndData(n):
    return jsonify({
        "received" : n,
        "sent" : n
    })

@app.route('/news')
def sendNews():
    return "aaj ki taaza khabar"

if __name__ == "__main__":
    app.run(debug=True)