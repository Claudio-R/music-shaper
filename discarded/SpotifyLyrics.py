import base64
from requests import post,get
import json

client_id ='cc1298c00eee4bf8a344bfb2056ac496'
client_secret = '27d6b03a2d8048e2bb48fe2d55d78aae'

def get_token():
    auth_string = client_id + ":" + client_secret
    auth_bytes = auth_string.encode("utf-8")
    auth_base64 = str(base64.b64encode(auth_bytes), "utf-8")

    url = "https://accounts.spotify.com/api/token"
    headers = {
        "Authorization":"Basic " + auth_base64,
        "Content-Type": "application/x-www-form-urlencoded"
    }

    data = {"grant_type": "client_credentials"}
    result = post(url, headers = headers, data = data)
    json_result = json.loads(result.content)
    token = json_result["access_token"]
    return token


#define authorization header for future requests
def get_auth_header(token): 
    return{"Authorization": "Bearer "+ token}

def search_for_song(token, artist_name, song_name):
    url = 'https://api.spotify.com/v1/search'
    headers = get_auth_header(token)
    full = song_name +" "+artist_name
    query = f"?q={full}&type=track&limit=1"
    #limit= 1 : gives only the first result ! 
    query_url = url + query
    result = get(query_url, headers = headers)
    json_result = json.loads(result.content)
    #Printing the artists 
    print("Artist(s):")
    for item in json_result["tracks"]["items"][0]["artists"]:
        print(item["name"])
    print("Title:")
    print(json_result["tracks"]["items"][0]["name"])
    print("Is it what you were looking for?")
    if len(json_result) == 0: 
        print("No artist with this name exists")
        return None
    return json_result["tracks"]["items"][0]["id"]


def search_for_songitems(artist_name, song_name):
    token = get_token()
    url = 'https://api.spotify.com/v1/search'
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
    token = get_token()
    id = search_for_song(token, artist, song)
    url = f"https://api.spotify.com/v1/audio-features/{id}"
    headers = get_auth_header(token)
    result = get(url, headers=headers)
    json_result = json.loads(result.content)
    tempo = json_result["tempo"]
    time_signature = json_result["time_signature"]
    return tempo, time_signature

    #useless, we can get playlist with playlists id
def get_users_playlists(token, user_id): 
    url = f"https://api.spotify.com/v1/users/{user_id}/playlists"
    headers = get_auth_header(token)
    result = get(url, headers=headers)
    json_result = json.loads(result.content)
    return json_result

def get_playlist_tracks(token, playlist_id):
    url = f"https://api.spotify.com/v1/playlists/{playlist_id}/tracks"
    headers = get_auth_header(token)
    result = get(url, headers=headers)
    json_result = json.loads(result.content)
    return json_result

def search_for_artist(token, artist_name):
    url = "https://api.spotify.com/v1/search"
    headers = get_auth_header(token)
    query = f"?q={artist_name}&type=artist&limit=1"
    query_url =url + query
    result= get(query_url, headers=headers)
    json_result= json.loads(result.content)["artists"]["items"]
    if len(json_result)==0:
        print("No artist with this name")
        return None
    return json_result[0]

#AQBrzklc4elr8yh9O7KmGEsN8yLwaq1BQIN-5oqllG0m1ZZUgtGliMsRXhpHcPm4IhYTB4NL8HyZ1JWBg2HM6Zt8RDUEXzgSWytzj4jeAH9wKuxL4iiaC1K-7J_AyZZei3I-ZavpfMfhDrDY2sFapF4YOlKyjmOS

#FINDING SP_DC: https://github.com/akashrchandran/syrics/wiki/Finding-sp_dc 


if __name__ == '__main__':
    from syrics.api import Spotify
    sp_dc = "AQBrzklc4elr8yh9O7KmGEsN8yLwaq1BQIN-5oqllG0m1ZZUgtGliMsRXhpHcPm4IhYTB4NL8HyZ1JWBg2HM6Zt8RDUEXzgSWytzj4jeAH9wKuxL4iiaC1K-7J_AyZZei3I-ZavpfMfhDrDY2sFapF4YOlKyjmOS"
    sp = Spotify(sp_dc)
