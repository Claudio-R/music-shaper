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

app = Flask(__name__, template_folder='client/template', static_folder='client/static')
run_with_ngrok(app)


@app.route("/")

@app.route('/home')
def home():
    return render_template('home.html')

@app.route('/submit', methods=['POST'])
def execute_script():
    try:
        config = request.get_json()
        process = multiprocessing.Process(target=generate_clip, args=(config,))
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
        time.sleep(15)
    try:
        return send_file(video_path, mimetype='video/mp4')
    except Exception as e:
        return jsonify({
            'message': 'Error while getting video',
            'status': 500
            })
        