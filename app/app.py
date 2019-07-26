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
    image_url0=assign_if_exist(0, urls),
    image_url1=assign_if_exist(1, urls),
    image_url2=assign_if_exist(2, urls),
    image_url3=assign_if_exist(3, urls),
    image_url4=assign_if_exist(4, urls),
    image_url5=assign_if_exist(5, urls),
    image_url6=assign_if_exist(6, urls),
    image_url7=assign_if_exist(7, urls))

def assign_if_exist(ind:0, urls:list):
    if ind < len(urls):
        return urls[ind]
    return 'https://cdn.pixabay.com/photo/2015/12/22/04/00/photo-1103595_640.png'

app.run(debug=True)

#TO VIEW WEBSITE< RUN THIS PROGRAM AND THEN PUT THIS IN YOUR NAV BAR:
#http://localhost:5000