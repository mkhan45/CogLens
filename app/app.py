import sys
from flask import *

sys.path.append('/Users/crystal/repositories/CogLens')
from main import search

app = Flask(__name__)

@app.route('/', methods=["GET", "POST"])
def index():
    return render_template('website.html')

@app.route('/submit', methods=["GET", "POST"])
def submit():
    if request.method == "GET":
        user_input = request.args["bar"]
        urls = search(str(user_input))
    return render_template('website.html',
    image_url0=urls[0],
    image_url1=urls[1],
    image_url2=urls[2],
    image_url3=urls[3],
    image_url4=urls[4],
    image_url5=urls[5],
    image_url6=urls[6],
    image_url7=urls[7])

app.run(debug=True)

#TO VIEW WEBSITE< RUN THIS PROGRAM AND THEN PUT THIS IN YOUR NAV BAR:
#http://localhost:5000