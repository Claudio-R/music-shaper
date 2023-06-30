import base64
from requests import post, get
import yaml
import json

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

def search_for_song_id(artist_name, song_name):
    url = 'https://api.spotify.com/v1/search'

    token = get_spotify_token()
    headers = get_auth_header(token)
    full = f'{song_name} {artist_name}'
    query = f'?q={full}&type=track&limit=1'
    
    query_url = url + query
    result = get(query_url, headers = headers)
    json_result = json.loads(result.content)
    if len(json_result) == 0: 
        print("No artist with this name exists")
        return None
    song_ID = json_result["tracks"]["items"][0]["id"]
    return song_ID

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
        print("Cannot find song")
        return None

    print("Retrieved song from Spotify:")
    
    print("Artist(s):")
    for item in json_result["tracks"]["items"][0]["artists"]:
        print(item["name"])

    print("Title:", json_result["tracks"]["items"][0]["name"])
    print("ID:", json_result["tracks"]["items"][0]["id"])

    return json_result["tracks"]["items"][0]["id"]


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
    try:
        id = search_for_song_id(artist, song)
        spotify_lyrics = sp_dc.get_lyrics(id)
    except Exception as e:
        print("No lyrics found on spotify")
        raise e

    lyrics = []
    for line in spotify_lyrics['lyrics']['lines']:
        lyrics.append({'text' : line['words'], 'start' : str(line['startTimeMs']/1000.0) })
    print("Lyrics found on Spotify!")
    return lyrics

if __name__=='__main__':
    spotify_token = get_spotify_token()
    print("token:", spotify_token)
    song_id = search_for_song_id('Drake', 'Gods Plan')
    print("song_id:", song_id)
    features = get_audio_features_given_song_id(song_id)
    print("features:", features) 