import requests
from youtube_transcript_api import YouTubeTranscriptApi
import yt_dlp as youtube_dl

def search_song_on_yt(api_key, artist, title, needLyrics = True):
    # The query to search for on YouTube
    query = artist + " " + title + " lyrics"
    captionsArg = ""
    print(query)

    # The URL to search for videoms on YouTube
    if needLyrics:
        captionsArg = "&videoCaption=closedCaption"

    url = f"https://www.googleapis.com/youtube/v3/search?part=snippet&q={query}&type=video{captionsArg}&key={api_key}"
    print("URL: ", url)

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
    print("Video ID:", video_id)
    print("Title:", video_title)
    print("Channel:", video_channel)

    return video_id

def search_lyrics_on_youtube(yt_video_id):
    lyrics = ""
    try:
        lyrics = YouTubeTranscriptApi.get_transcript(yt_video_id, languages=['en', 'en-US'])
    except Exception as e:
        print(f"Error: {e}")

    return lyrics

def downloadSongFromYT(url, outPath):
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
            return True
    except Exception as e:
        print(e)
        return False 

