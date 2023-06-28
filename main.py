from source.clip_generation import generate_clip
from server import app

#TODO - Fondere le due post createClip e get video
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