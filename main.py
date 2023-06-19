import yaml
import sys

#TODO: the main should call the gui, then the gui should call everything else

with open("env.local.yml", "r") as f:
    credentials = yaml.safe_load(f)
    sp_dc = credentials["SP_DC"]

if __name__ == "__main__":

    sys.path.append("source")

    from source.spotify_client import get_spotify_token, search_for_song_id, get_audio_features_given_song_id
    from source.mood_prediction import moods_prediction
    from syrics.api import Spotify

    spotify_token  = get_spotify_token()
    artist = "Thirty Seconds To Mars"
    song = "The Kill"
    song_id = search_for_song_id(spotify_token, artist, song)
    features = get_audio_features_given_song_id(spotify_token, song_id)

    sp = Spotify(sp_dc)
    lrc= sp.get_lyrics(song_id)
    lyrics = []
    for line in range(len(lrc["lyrics"]['lines'])): 
        time = int(lrc["lyrics"]['lines'][line]['startTimeMs'])/1000.0
        words = lrc["lyrics"]['lines'][line]['words']
        lyrics.append({'text' : words, 'start' : str(time) })
    
    for item in lyrics:
        print(item)

    print("Mood: ", moods_prediction(spotify_token, song_id))