import yaml
from source.clip_generation import generate_clip
from server.server import app

with open("env.local.yml", "r") as f:
    credentials = yaml.safe_load(f)
    sp_dc = credentials["SP_DC"]

class MusicShaper():
    def __init__(self) -> None:
        pass

    def generate_clip(self) -> None:
        artist = input("Insert artist: ")
        song = input("Insert song: ")
        generate_clip(artist, song)

    def run_gui(self, app) -> None:
        app.run()
    

if __name__ == "__main__":
    @app.route('/execute_script', methods=['POST'])
    def execute_script():
        print("sono entrato in execute_script")
        generate_clip("", "")

    music_shaper = MusicShaper()
    music_shaper.generate_clip()
    # music_shaper.run_gui(app)