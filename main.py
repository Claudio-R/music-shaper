from source.clip_generation import generate_clip
from server import app
from flask import jsonify

@app.route('/execute_script', methods=['POST'])
def execute_script():
    print("sono entrato in execute_script")
    try:
        generate_clip(None)
        return jsonify({
            'message': 'Script eseguito correttamente',
            'status': 200
            })
    except Exception as e:
        print(e)
        return jsonify({
            'message': 'Errore durante l\'esecuzione dello script',
            'status': 500
            })

class MusicShaper():
    def __init__(self) -> None:
        pass

    def generate_video(self) -> None:
        config = {
            'artist': input("Insert artist: "),
            'song': input("Insert song: "),
            'style1': input("Insert main style: "),
            'style2': input("Insert secondary style: "),
            'content': input("Insert content: ")
        }
        generate_clip(config)

    def run_gui(self) -> None:
        global app
        app.run()