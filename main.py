import yaml
from source.clip_generation import generate_clip
from server.server import app

with open("env.local.yml", "r") as f:
    credentials = yaml.safe_load(f)
    sp_dc = credentials["SP_DC"]

@app.route('/execute_script', methods=['POST'])
def execute_script():
    print("sono entrato in execute_script")
    generate_clip("", "")

class MusicShaper():
    def __init__(self) -> None:
        pass

    def generate_clip(self) -> None:
        artist = input("Insert artist: ")
        song = input("Insert song: ")
        generate_clip(artist, song)

    def run_gui(self, app) -> None:
        app.run()
    

def main():
    music_shaper = MusicShaper()

    #TODO - separate setup from execution
    # music_shaper.generate_clip()
    # music_shaper.run_gui(app)

    return music_shaper