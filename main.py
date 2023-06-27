from source.clip_generation import generate_clip
from server.server import app

@app.route('/execute_script', methods=['POST'])
def execute_script():
    print("sono entrato in execute_script")
    generate_clip(None)
    return "ok"

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