from flask import *

app = Flask(__name__)

@app.route('/', methods=["GET", "POST"])
def index():
    return render_template('website.html')

@app.route('/submit', methods=["GET", "POST"])
def submit():
    if request.method == "GET":
        input = request.args["bar"]
        print(input)
    return render_template('website.html')

app.run(debug=True)

#TO VIEW WEBSITE< RUN THIS PROGRAM AND THEN PUT THIS IN YOUR NAV BAR:
#http://localhost:5000