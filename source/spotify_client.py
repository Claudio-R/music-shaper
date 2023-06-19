import base64
from requests import post, get
import yaml
import json

with open("env.local.yml", 'r') as stream:
    try:
        credentials = yaml.safe_load(stream)
        client_id = credentials['CLIENT_ID']
        client_secret = credentials['CLIENT_SECRET']
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

def search_for_song_id(token, artist_name, song_name):
    url = 'https://api.spotify.com/v1/search'
    
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

def get_audio_features_given_song_id(token, id): 
    url = f"https://api.spotify.com/v1/audio-features/{id}"
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

if __name__=='__main__':
    spotify_token = get_spotify_token()
    print("token:", spotify_token)
    song_id = search_for_song_id(spotify_token, 'Drake', 'Gods Plan')
    print("song_id:", song_id)
    features = get_audio_features_given_song_id(spotify_token, song_id)
    print("features:", features) 