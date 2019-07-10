# import the nessecary pieces from Flask
from flask import Flask,render_template, request,jsonify,Response
#Create the app object that will route our calls
app = Flask(__name__)
# Route the user to the homepage
@app.route('/', methods = ['GET'])
def home():
    return render_template('home.html')

#When run from command line, start the server
if __name__ == '__main__':
    app.run(host ='0.0.0.0', port = 3333, debug = True)