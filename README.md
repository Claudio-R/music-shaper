# music-shaper
AI-based Automatic Musical Videoclip Generation

## Requirements

## How to use
Open a colab notebook and clone this repository:
```
!git clone https://github.com/Claudio-R/music-shaper
%cd music-shaper/
```
Then, create a file named `env.local.yml` with the same entries as the file `env.example.yml`, and fill it with your own keys.


From the main script `main.py` import the `MusicShaper` class and create an instance of it:
```
from main import MusicShaper
music_shaper = MusicShaper()
```
It is possible to use the `music_shaper` object to generate your own videoclips directly from the command line with:
```
music_shaper.generate_video()
```
Otherwise, it is possible to start the gui with:
```
music_shaper.run_gui()
```

## How to get your own API keys
### Spotify
1. Go to https://developer.spotify.com/dashboard/login and login with your Spotify account
2. Click on "Create an app"
3. Fill the form with the name of your app and a description
4. Click on "Edit settings" and add `http://localhost:8080` as a redirect URI
5. Click on "Show client secret" and copy the *Client ID* and the *Client Secret* in your `env.local.yml` file
6. To get your `sp_dc` token, follow the instructions at https://github.com/akashrchandran/syrics/wiki/Finding-sp_dc.

### OpenAI
1. Go to https://beta.openai.com/ and login with your OpenAI account
2. Click on "API Keys" and then on "Create new API key"
3. Copy the API key in your `env.local.yml` file

### Youtube
1. Go to https://console.developers.google.com/ and login with your Google account
2. Click on "Create project"
3. Click on "Enable APIs and services"
4. Search for "YouTube Data API v3" and click on "Enable"

### NGROK
1. Go to https://dashboard.ngrok.com/get-started/setup and follow the instructions to install NGROK

