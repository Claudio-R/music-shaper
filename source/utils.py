from __future__ import unicode_literals

import librosa
import soundfile as sf
import numpy as np
import random

# TODO: 
from source.spotify_api import search_lyrics_on_spotify
from source.youtube_api import search_lyrics_on_youtube

def print_lyrics(lyrics):
    print("Requested lyrics:")
    if type(lyrics) == str:
        print(lyrics)
    else:
        for item in lyrics:
            print(item)

def import_lyrics_from_file():
    return ""

def get_lyrics(artist, song):
    id = ""
    try:
        lyrics = search_lyrics_on_spotify(artist, song)
    except:
        try:
            id, lyrics = search_lyrics_on_youtube(artist, song)
        except Exception as e:
            print("No lyrics found")
            raise e                 
    return (id, lyrics)

def cutAudio(path, outPath, start, end): 
    y, sr = librosa.load(path)
    start_samples = int(start*sr)
    end_samples = int(end*sr)
    y_cut = y[start_samples : end_samples]
    
    sf.write(outPath, y_cut,  sr)
    return y_cut

def get_onsets_2d(path, sr, norm): 
    y, sr = librosa.load(path, sr=sr)
    y_n = (y-y.min())/(y.max()-y.min())*(1) + 1
    max = y_n.max()

    for i in range(len(y_n)): 
      if(y_n[i] < max-norm): 
          y_n[i] = 1.0

    peaks = np.zeros(len(y_n))
    for i in range(len(peaks)): 
        if(y_n[i] != 1.0): 
            peaks[i] = 1

    sd_list  = ""
    sd_list_n = ""
    for i in range(len(y_n)) :
        
        if(i != len(y_n) -1  ): 
            sd_list = sd_list + str(i) + ":(" + str(y_n[i]) + "), "
        else: 
            sd_list = sd_list + str(i) + ":(" + str(y_n[i]) + ")"
  
    for i in range(len(y_n)) :
      
        if(i != len(y_n) -1  ): 
            if(peaks[i+1] == 1.0 and i < len(y_n) - 2 ): 
                sd_list_n = sd_list_n + str(i) + ":(" + str(y_n[i+1]-0.1) + "), "
            else:
                sd_list_n = sd_list_n + str(i) + ":(" + str(y_n[i]) + "), "

        else: 
            sd_list_n = sd_list_n + str(i) + ":(" + str(y_n[i]) + ")"
    
    return sd_list_n

def get_onsets_3d(path, sr, norm): 
  y, sr = librosa.load(path, sr=sr)
  
  y_n = (y-y.min())/(y.max()-y.min())*(10)
  max = y_n.max()
  
  for i in range(len(y_n)): 
    if(y_n[i] < max-norm): 
      y_n[i] = 0

  peaks = np.zeros(len(y_n))
  for i in range(len(peaks)): 
    if(y_n[i] != 0): 
      peaks[i] = 1
  sd_list  = ""
  sd_list_n = ""
  for i in range(len(y_n)) :
    
    if(i != len(y_n) -1  ): 
      sd_list = sd_list + str(i) + ":(" + str(y_n[i]) + "), "
    else: 
      sd_list = sd_list + str(i) + ":(" + str(y_n[i]) + ")"
  
  for i in range(len(y_n)) :
    
    if(i != len(y_n) -1  ): 
      if(peaks[i+1] == 1.0 and i < len(y_n) - 2 ): 
        sd_list_n = sd_list_n + str(i) + ":(" + str(y_n[i+1]-2) + "), "

      else:
        sd_list_n = sd_list_n + str(i) + ":(" + str(y_n[i]) + "), "

    else: 
      sd_list_n = sd_list_n + str(i) + ":(" + str(y_n[i]) + ")"

  return sd_list_n

def get_beats_librosa_zoom(path, sr, tempo, ts, min_zoom, max_zoom): 
    y, sr_x = librosa.load(path)
    tempo, beats =  librosa.beat.beat_track(y=y, sr=sr_x, bpm = tempo)
    beats_times = librosa.frames_to_time(beats)
    y_d, sr_d = librosa.load(path, sr = sr)
    ts_fs = 1/sr_d
  
    array_fs = np.arange(0, len(y_d)*ts_fs, ts_fs)
    new_beats = []
    tc = 0 

    for i in range(len(array_fs)): 
        if(i<len(array_fs) -2 ): 
            while((beats_times[tc] > array_fs[i] and beats_times[tc] < array_fs[i+1]) and (tc < beats_times.size - 1 )): 
                if((tc+1)%ts==0):
                    new_beats.append(i)
                tc = tc +1

    sd_list_n = ""
    beats_ar = np.full(len(array_fs), 1.0)

    for i in range(len(beats_ar)): 
        for b in new_beats:
            if(i == b): 
                beats_ar[i] = random.uniform(min_zoom, max_zoom)

    for i in range(len(beats_ar)):
        if(i != len(beats_ar) -1  ):     
            if(beats_ar[i+1] != 1.0 ): 
                sd_list_n = sd_list_n + str(i) + ":(" + str(beats_ar[i+1]-0.1) +"), "
            else: 
                sd_list_n = sd_list_n + str(i) + ":("+ str(beats_ar[i]) +"), "
        else: 
            sd_list_n = sd_list_n + str(i) + ":("+ str(beats_ar[i]) +")"
    
    return sd_list_n


def get_beats_librosa_angle(path, sr, tempo, ts, min_angle, max_angle): 
    y, sr_x = librosa.load(path)
    tempo, beats =  librosa.beat.beat_track(y=y, sr=sr_x, bpm = tempo)
    beats_times = librosa.frames_to_time(beats)
  
    y_d, sr_d = librosa.load(path, sr = sr)
  
    ts_fs = 1/sr_d
  
    array_fs = np.arange(0, len(y_d)*ts_fs, ts_fs)
    new_beats = []
    tc = 0 

    for i in range(len(array_fs)): 
        if(i<len(array_fs) -2 ): 
            while((beats_times[tc] > array_fs[i] and beats_times[tc] < array_fs[i+1]) and (tc < beats_times.size - 1 )): 
                if((tc+1)%ts==0):
                    new_beats.append(i)
                tc = tc +1
    sd_list_n = ""
    beats_ar = np.full(len(array_fs), 0)

    for i in range(len(beats_ar)): 
        for b in new_beats:
            if(i == b): 
                beats_ar[i] = random.randint(min_angle, max_angle)
    
    for i in range(len(beats_ar)) :
        if(i != len(beats_ar) -1  ): 
      
            sd_list_n = sd_list_n + str(i) + ":("+ str(beats_ar[i]) +"), "
        else: 
            sd_list_n = sd_list_n + str(i) + ":("+ str(beats_ar[i]) +")"
    
    return sd_list_n

if __name__ == "__main__":
    lyrics = get_lyrics("pino daniele", "ma che mania")
    print(lyrics)