from flask.templating import render_template
from flask import Flask, request, jsonify, send_file
import os, time, yaml
from flask_ngrok import run_with_ngrok

template_folder = '../gui/template'
static_folder = '../gui/static'

with open('env.local.yml') as f:
    env = yaml.load(f, Loader=yaml.FullLoader)
    cmd = 'ngrok authtoken ' + env['NGROK_AUTHORIZATION_TOKEN']
    os.system(cmd)
    
app = Flask(__name__, template_folder=template_folder, static_folder=static_folder)
run_with_ngrok(app)

@app.route("/")

@app.route('/home')
def home():
    return render_template('home.html')

@app.route('/process_array', methods=['POST'])
def process_array():
    data = request.get_json()  # ottieni i dati JSON dal corpo della richiesta
    array_data = data.get('arrayData')  # ottieni la stringa dal campo "stringData"
    print(array_data)
    response = {'message': 'Array_names ricevuto correttamente'}

    with open("./database/names.txt", "w") as fo:
        for i in range(0, len(array_data)):
            print(array_data[i]),
            print(array_data[i])
            if i == 0:
                fo.write(array_data[i] + "&")
            else:
                fo.write(array_data[i])
    return jsonify(response)

@app.route('/process_array_choices', methods=['POST'])
def process_array_choices():
    data_choices = request.get_json()  # ottieni i dati JSON dal corpo della richiesta
    array_data_choices = data_choices.get('arrayDataChoices')  # ottieni la stringa dal campo "stringData"

    response = {'message': 'Array_choices ricevuto correttamente'}

    with open("./database/style.txt", "w") as fo:
        for j in range(0, len(array_data_choices)):
            print(array_data_choices[j]),
            if j == 0 or j == 1:
                fo.write(array_data_choices[j] + "&")
            else:
                fo.write(array_data_choices[j])
    return jsonify(response)

@app.route('/video')
def get_video():
    print("sono entrato in get_video")
    video_path = './AI/Video/Music_cut.mp4'

    while not os.path.exists(video_path):
        # Attendi 1 secondo prima di riprovare
        time.sleep(10)

    return send_file(video_path, mimetype='video/mp4')
