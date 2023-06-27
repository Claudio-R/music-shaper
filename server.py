from flask.templating import render_template
from flask import Flask, request, jsonify, send_file
import os, time, yaml
from flask_ngrok import run_with_ngrok
from source.clip_generation import generate_clip

try:
    with open('env.local.yml') as f:
        try:
            env = yaml.load(f, Loader=yaml.FullLoader)
            cmd = 'ngrok authtoken ' + env['NGROK_AUTHORIZATION_TOKEN']
            os.system(cmd)
        except yaml.YAMLError as exc:
            print(exc)
except FileNotFoundError:
    input("Insert a valid env.local.yml file and press enter...")
    with open('env.local.yml') as f:
        try:
            env = yaml.load(f, Loader=yaml.FullLoader)
            cmd = 'ngrok authtoken ' + env['NGROK_AUTHORIZATION_TOKEN']
            os.system(cmd)
        except yaml.YAMLError as exc:
            print(exc)

app = Flask(__name__, template_folder='gui/template', static_folder='gui/static')
run_with_ngrok(app)

@app.route("/")

@app.route('/home')
def home():
    return render_template('home.html')

@app.route('/process_artist_song', methods=['POST'])
def process_array():
    data = request.get_json()
    array_data = data.get('arrayData')

    try:
        with open("./database/names.txt", "w") as fo:
            for i in range(0, len(array_data)):
                print(array_data[i])
                if i == 0:
                    fo.write(array_data[i] + "&")
                else:
                    fo.write(array_data[i])
        response = jsonify({
            'message': 'Array_names retrieve successfully',
            'status': 200
            })
    except Exception as e:
        print(e)
        response = jsonify({
            'message': 'Error while retrieving array_names',
            'status': 500
            })
    
    print(response)
    return response

@app.route('/process_style_content', methods=['POST'])
def process_array_choices():
    data_choices = request.get_json()
    array_data_choices = data_choices.get('arrayDataChoices')
    
    try:
        with open("./database/style.txt", "w") as fo:
            for j in range(0, len(array_data_choices)):
                print(array_data_choices[j]),
                if j == 0 or j == 1:
                    fo.write(array_data_choices[j] + "&")
                else:
                    fo.write(array_data_choices[j])

        response = jsonify({
            'message': 'Array_styles retrieve successfully',
            'status': 200
            })
    except Exception as e:
        print(e)
        response = jsonify({
            'message': 'Error while retrieving array_styles',
            'status': 500
            })  
        
    print(response)
    return response      

@app.route('/execute_script', methods=['POST'])
def execute_script():
    try:
        response = jsonify({
            'message': 'Generating clip...',
            'status': 200,
            })
        generate_clip(None)
    except Exception as e:
        print(e)
        response = jsonify({
            'message': 'Error while generating clip',
            'status': 500
            })
        
    print(response)
    return response

@app.route('/get_video')
def get_video():
    video_path = 'AI/Video/Music_cut.mp4'
    while not os.path.exists(video_path):
        print("Path does not exist")
        time.sleep(10)

    try:
        send_file(video_path, mimetype='video/mp4')
        response = jsonify({
            'message': 'Video retrieved successfully',
            'status': 200
            })
    except Exception as e:
        print(e)
        response = jsonify({
            'message': 'Error while retrieving video',
            'status': 500
            })
        
    print(response)
    return response