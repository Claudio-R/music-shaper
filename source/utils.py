from __future__ import unicode_literals

import librosa
import soundfile as sf
import numpy as np
import random

from source.spotify_api import search_lyrics_on_spotify
from source.youtube_api import search_song_on_yt, search_lyrics_on_youtube

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
    try:
        lyrics = search_lyrics_on_spotify(artist, song)
    except:
        print("Exception Occurred in search_on_spotify function: The song has no lyrics or doesn't exist")
        lyrics=[]

    if not lyrics:
        id = search_song_on_yt(artist, song)
        lyrics = search_lyrics_on_youtube(id)

    return lyrics

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
  #print('peaks', peaks)
  #0.5 delta massimo
  #fact, decide il range
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

# -------------------------------------------------------------------------------------------NOT USED
def get_onsets_angles(path, sr, norm, delta): 
  y, sr = librosa.load(path, sr=sr)
  
  y_n = (y-y.min())/(y.max()-y.min())*(-2*delta) + delta
  max = abs(y_n).max()
  #print(y_n)
  #print('Max', y_n.max())
  #print('Min', y_n.min())
  #print('Len', len(y_n))
  
  for i in range(len(y_n)): 
    if(abs(y_n[i]) < max-norm ): 
      y_n[i] = 0

  peaks = np.zeros(len(y_n))
  for i in range(len(peaks)): 
    if(y_n[i] != 0): 
      peaks[i] = 1
  #print('peaks', peaks)
  #0.5 delta massimo
  #fact, decide il range
  sd_list  = ""
  sd_list_n = ""
  for i in range(len(y_n)) :
    
    if(i != len(y_n) -1  ): 
      sd_list = sd_list + str(i) + ":(" + str(y_n[i]) + "), "
    else: 
      sd_list = sd_list + str(i) + ":(" + str(y_n[i]) + ")"
  
  #print('before\n', sd_list)
  for i in range(len(y_n)) :
    
    if(i != len(y_n) -1  ): 
      if(peaks[i+1] == 1.0 and i < len(y_n) - 2 ): 
   #     print('check',i, y_n[i+1]-0.1 )
        sd_list_n = sd_list_n + str(i) + ":(" + str(y_n[i+1]-2) + "), "

      else:
        sd_list_n = sd_list_n + str(i) + ":(" + str(y_n[i]) + "), "

    else: 
      sd_list_n = sd_list_n + str(i) + ":(" + str(y_n[i]) + ")"
    
  
  #print('after\n', sd_list_n)
  
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
                #NOTE - MIN:MAX ZOOM
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
                #NOTE - MIN:MAX ANGLE
                beats_ar[i] = random.randint(min_angle, max_angle)
    
    for i in range(len(beats_ar)) :
        if(i != len(beats_ar) -1  ): 
      
            sd_list_n = sd_list_n + str(i) + ":("+ str(beats_ar[i]) +"), "
        else: 
            sd_list_n = sd_list_n + str(i) + ":("+ str(beats_ar[i]) +")"
    
    return sd_list_n

