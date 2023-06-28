from flask.templating import render_template
from flask import Flask, request, jsonify, send_file
import os, time, yaml, multiprocessing
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

def generate_video(config):
    try:
        generate_clip(config)
        print('video_ready', {'message': 'Video has been correctly generated'})
    except Exception as e:
        print(e)
        print('video_error', {'message': 'Error while generating video'})

@app.route("/")

@app.route('/home')
def home():
    return render_template('home.html')
   
@app.route('/submit', methods=['POST'])
def execute_script():
    try:
        config = request.get_json()
        process = multiprocessing.Process(target=generate_video, args=(config,))
        process.start()
        response = jsonify({
            'message': 'Generating clip...',
            'status': 200,
            })

    except Exception as e:
        print(e)
        response = jsonify({
            'message': 'Error while generating clip',
            'status': 500
            })
        
    return response

@app.route('/get_video')
def get_video():
    video_path = 'AI/Video/Music_cut.mp4'
    while not os.path.exists(video_path):
        print("Path to video does not exist")
        time.sleep(10)

    try:
        return send_file(video_path, mimetype='video/mp4')
    except Exception as e:
        print(e)