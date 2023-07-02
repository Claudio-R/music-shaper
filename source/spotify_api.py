import base64
from requests import post, get
import yaml
import json
from syrics.api import Spotify

try:
    with open("env.local.yml", 'r') as stream:
        try:
            credentials = yaml.safe_load(stream)
            client_id = credentials['CLIENT_ID']
            client_secret = credentials['CLIENT_SECRET']
            sp_dc = credentials['SP_DC']
        except yaml.YAMLError as exc:
            print(exc)
except FileNotFoundError:
    input("Insert a valid env.local.yml file and press enter...\n")
    with open("env.local.yml", 'r') as stream:
        try:
            credentials = yaml.safe_load(stream)
            client_id = credentials['CLIENT_ID']
            client_secret = credentials['CLIENT_SECRET']
            sp_dc = credentials['SP_DC']
        except yaml.YAMLError as exc:
            print(exc)
    
def get_spotify_token() -> str:
    auth_string = f"{client_id}:{client_secret}"
    auth_bytes = auth_string.encode("utf-8")
    auth_base64 = str(base64.b64encode(auth_bytes), "utf-8")

    url = "https://accounts.spotify.com/api/token"
    headers = {
        "Authorization": f"Basic {auth_base64}",
        "Content-Type": "application/x-www-form-urlencoded"
    }
    data = { "grant_type": "client_credentials" }
    
    response = post(url, data=data, headers=headers)
    json_response = json.loads(response.content)
    token = json_response['access_token']
    return token

def get_auth_header(token): 
    return{'Authorization': f'Bearer {token}'}

def get_audio_features_given_song_id(id): 
    url = f"https://api.spotify.com/v1/audio-features/{id}"
    token = get_spotify_token()
    headers = get_auth_header(token)
    result = get(url, headers=headers)
    json_result = json.loads(result.content)
    length = json_result["duration_ms"]
    danceability = json_result["danceability"]
    acousticness = json_result["acousticness"]
    energy = json_result["energy"]
    instrumentalness = json_result["instrumentalness"]
    liveness= json_result["liveness"]
    valence = json_result["valence"]
    loudness = json_result["loudness"]
    spechiness = json_result["speechiness"]
    tempo = json_result["tempo"]

    return [length, danceability, acousticness, energy, instrumentalness, liveness, valence, loudness, spechiness, tempo]

def search_for_song_id(artist_name, song_name):
    url = 'https://api.spotify.com/v1/search'
    token = get_spotify_token()
    headers = get_auth_header(token)
    full = song_name +" "+artist_name
    query = f"?q={full}&type=track&limit=1"

    query_url = url + query
    result = get(query_url, headers = headers)

    json_result = json.loads(result.content)

    if len(json_result) == 0:
        raise Exception("Cannot find song")
    else:
        print("Song found on spotify!")

    sp_ID = json_result["tracks"]["items"][0]["id"]
    print("Artist(s): ", json_result["tracks"]["items"][0]["artists"][0]["name"])
    print("Title:", json_result["tracks"]["items"][0]["name"])
    print("Spotify ID:", sp_ID)

    return sp_ID


def search_for_songitems(artist_name, song_name):
    url = 'https://api.spotify.com/v1/search'
    token = get_spotify_token()
    headers = get_auth_header(token)
    full = song_name +" "+artist_name
    query = f"?q={full}&type=track&limit=1"
    query_url = url + query
    result = get(query_url, headers = headers)
    json_result = json.loads(result.content)
    if len(json_result) == 0: 
        print("No artist with this name exists")
        return None
    return json_result["tracks"]["items"]


#taking audio features given id of song
def get_tempo_ts(artist, song): 
    token = get_spotify_token()
    id = search_for_song_id(artist, song)
    url = f"https://api.spotify.com/v1/audio-features/{id}"
    headers = get_auth_header(token)
    result = get(url, headers=headers)
    json_result = json.loads(result.content)
    tempo = json_result["tempo"]
    time_signature = json_result["time_signature"]
    return tempo, time_signature

    #useless, we can get playlist with playlists id
def get_users_playlists(user_id): 
    url = f"https://api.spotify.com/v1/users/{user_id}/playlists"
    token = get_spotify_token()
    headers = get_auth_header(token)
    result = get(url, headers=headers)
    json_result = json.loads(result.content)
    return json_result

def get_playlist_tracks(playlist_id):
    url = f"https://api.spotify.com/v1/playlists/{playlist_id}/tracks"
    token = get_spotify_token()
    headers = get_auth_header(token)
    result = get(url, headers=headers)
    json_result = json.loads(result.content)
    return json_result

def search_for_artist(artist_name):
    url = "https://api.spotify.com/v1/search"
    token = get_spotify_token()
    headers = get_auth_header(token)
    query = f"?q={artist_name}&type=artist&limit=1"
    query_url =url + query
    result= get(query_url, headers=headers)
    json_result= json.loads(result.content)["artists"]["items"]
    if len(json_result)==0:
        print("No artist with this name")
        return None
    return json_result[0]

def search_lyrics_on_spotify(artist, song): 
    print("\nSearching for lyrics on spotify...")
    id = search_for_song_id(artist, song)
    sp = Spotify(sp_dc)
    
    print("Retrieving lyrics...")
    spotify_lyrics = sp.get_lyrics(id)
    lyrics = []

    for line in range(len(spotify_lyrics["lyrics"]['lines'])): 
        time = int(spotify_lyrics["lyrics"]['lines'][line]['startTimeMs'])/1000.0
        words = spotify_lyrics["lyrics"]['lines'][line]['words']
        lyrics.append({'text' : words, 'start' : str(time) })
    print("My lyrics:", lyrics)
    
    return lyrics

if __name__=='__main__':
    spotify_token = get_spotify_token()
    print("token:", spotify_token)
    song_id = search_for_song_id('Drake', 'Gods Plan')
    print("song_id:", song_id)
    features = get_audio_features_given_song_id(song_id)
    print("features:", features) 