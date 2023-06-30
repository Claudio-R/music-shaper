import requests, yaml
from youtube_transcript_api import YouTubeTranscriptApi
import yt_dlp as youtube_dl

try:
    with open("env.local.yml", "r") as f:
        try:
            credentials = yaml.safe_load(f)
            youtube_api_key = credentials["YOUTUBE_DATA_API_V3_KEY"]
        except yaml.YAMLError as exc:
            print(exc)
except FileNotFoundError:
    input("Insert a valid env.local.yml file and press enter...\n")
    with open("env.local.yml", "r") as f:
        try:
            credentials = yaml.safe_load(f)
            youtube_api_key = credentials["YOUTUBE_DATA_API_V3_KEY"]
        except yaml.YAMLError as exc:
            print(exc)

def search_song_on_yt(artist, title, needLyrics=False):
    # The query to search for on YouTube
    query = artist + "+" + title + "+lyrics"
    captionsArg = ""

    # The URL to search for videoms on YouTube
    if needLyrics:
        captionsArg = "&videoCaption=closedCaption"

    url = f"https://www.googleapis.com/youtube/v3/search?part=snippet&q={query}&type=video{captionsArg}&key={youtube_api_key}"

    # Perform the search
    response = requests.get(url).json()

    #print(response)
    # Get the first video from the search results
    video = response["items"][0]

    # Get the video information
    video_id = video["id"]["videoId"]
    video_title = video["snippet"]["title"]
    video_channel = video["snippet"]["channelTitle"]

    # Print the video information
    print("Song found on YouTube!")
    print("Video ID:", video_id)
    print("Title:", video_title)
    print("Channel:", video_channel)

    return video_id

def search_lyrics_on_youtube(artist, song):
    print("Searching for lyrics on YouTube...")
    try:
        id = search_song_on_yt(artist, song, True)
        lyrics = YouTubeTranscriptApi.get_transcript(id, languages=['en', 'en-US'])
    except Exception as e:
        print("No lyrics found on YouTube")
        raise e
    return lyrics

# def download_song(url, outPath):
def download_song(artist, song, outPath):
    id = search_song_on_yt(artist, song)
    url = 'https://www.youtube.com/watch?v=' + id
    ydl_opts = {
        'format': 'bestaudio/best',
        'outtmpl': outPath,
        'noplaylist': True,
        'continue_dl': True,
        'postprocessors': [{
            'key': 'FFmpegExtractAudio',
            'preferredcodec': 'wav',
            'preferredquality': '192', }]
    }
    try:
        with youtube_dl.YoutubeDL(ydl_opts) as ydl:
            ydl.cache.remove()
            ydl.download([url])
    except Exception as e:
        print("Error downloading song from YouTube")
        raise e
