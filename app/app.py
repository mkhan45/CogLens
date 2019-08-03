import sys, pickle
from flask import *
from time import sleep

sys.path.append('/Users/crystal/repositories/CogLens')
from main import search

sys.path.append('/Users/crystal/repositories/Gucci-Group-Musicthing')
from input_audio import get_mic_data, read_database_file
from check_matches import check_matches

sys.path.append('/Users/crystal/repositories/textGen')
from text_gen import *

app = Flask(__name__)

#INIT
@app.route('/', methods=["GET", "POST"])
def index():
    return render_template('song_rec.html', record_text="Recording...")

#NAV
@app.route('/goto_song', methods=["GET", "POST"])
def goto_song():
    return index()

@app.route('/goto_text', methods=["GET", "POST"])
def goto_text():
    return render_template('text_gen.html')

@app.route('/goto_search', methods=["GET", "POST"])
def goto_search():
    return render_template('img_search.html')

#SONG REC
@app.route('/submit_song', methods=['GET', 'POST'])
def submit_song_database():
    database_name = request.args['song_database_bar']
    database = read_database_file(str(database_name))
    seconds = float(request.args['time_bar'])
    mic_data = get_mic_data(seconds)

    matches = check_matches(mic_data, database.dictionary)

    if matches != -1:
        return render_template('song_rec.html', info_text=database.id_to_name[matches], record_text="")
    else:
        return render_template('song_rec.html', info_text='Song not recognized.', record_text="")

@app.route('/ok', methods=["GET", "POST"])
def ok():
    return render_template('song_rec.html', info_text="", record_text="Recording...")

#IMAGE SEARCH
@app.route('/submit_search', methods=["GET", "POST"])
def submit_search():
    if request.method == "GET":
        user_input = request.args["search_bar"]
        urls = search(str(user_input))
    return render_template('img_search.html',
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

#TEXT GEN
database1 = dict()

with open('/Users/crystal/repositories/textGen/text_model_11.pkl', 'rb') as f:
    lm11 = pickle.load(f) #lm11?
@app.route('/submit_num_words', methods=["GET", "POST"])
def submit_num_words():
    return render_template('text_gen.html', gen_text=generate_text(lm11, 11, 500))

app.run(debug=True)

#TO VIEW WEBSITE< RUN THIS PROGRAM AND THEN PUT THIS IN YOUR NAV BAR:
#http://localhost:5000
#OR, IF THAT DOESN'T WORK
#http://127.0.0.1:5000