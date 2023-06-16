
from __future__ import unicode_literals

import librosa
import yt_dlp as youtube_dl
import soundfile as sf
import numpy as np
import random

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

def get_beats_librosa_zoom(path, sr, tempo, ts): 
  y, sr_x = librosa.load(path)
  
  #onset_envelope = librosa.onset.onset_strength(y=y, sr=sr_x)
  
  #onsets = librosa.onset.onset_detect(onset_envelope=onset_envelope)
  tempo, beats =  librosa.beat.beat_track(y=y, sr=sr_x, bpm = tempo)
  beats_times = librosa.frames_to_time(beats)
  y_d, sr_d = librosa.load(path, sr = sr)
  ts_fs = 1/sr_d
  
  array_fs = np.arange(0, len(y_d)*ts_fs, ts_fs)
  #print('ar', array_fs)
  new_beats = []
  tc = 0 
  for i in range(len(array_fs)): 

    if(i<len(array_fs) -2 ): 

      while((beats_times[tc] > array_fs[i] and beats_times[tc] < array_fs[i+1]) and (tc < beats_times.size - 1 )): 
        if((tc+1)%ts==0):
          new_beats.append(i)
          #print('bt', beats_times[tc])
        #same values may be present
        #controllo su length di onset times !!!! 
          #if((beats_times[tc] > array_fs[i] + ts_fs/2) and (beats_times[tc] < array_fs[i] + ts_fs)): 
          #  new_beats.append(i)
          #elif((beats_times[tc] > array_fs[i]) and (beats_times[tc] < array_fs[i] + ts_fs/2)): 
          #  new_beats.append(i+1)
        tc = tc +1

  #print(new_beats)
  sd_list_n = ""
  beats_ar = np.full(len(array_fs), 1.0)

  for i in range(len(beats_ar)): 
    for b in new_beats:
      if(i == b): 
        beats_ar[i] = random.uniform(1.0, 2.0)

    for i in range(len(beats_ar)) :
        if(i != len(beats_ar) -1  ):       
            if(beats_ar[i+1] != 1.0 ): 
                sd_list_n = sd_list_n + str(i) + ":(" + str(beats_ar[i+1]-0.1) +"), "
            else: 
                sd_list_n = sd_list_n + str(i) + ":("+ str(beats_ar[i]) +"), "
        else: 
            sd_list_n = sd_list_n + str(i) + ":("+ str(beats_ar[i]) +")"

    return sd_list_n

def get_beats_librosa_angle(path, sr, tempo, ts): 
  y, sr_x = librosa.load(path)
  
  #onset_envelope = librosa.onset.onset_strength(y=y, sr=sr_x)
  
  #onsets = librosa.onset.onset_detect(onset_envelope=onset_envelope)
  tempo, beats =  librosa.beat.beat_track(y=y, sr=sr_x, bpm = tempo)
  beats_times = librosa.frames_to_time(beats)
  
  y_d, sr_d = librosa.load(path, sr = sr)
  
  ts_fs = 1/sr_d
  
  array_fs = np.arange(0, len(y_d)*ts_fs, ts_fs)
  #print('ar', array_fs)
  new_beats = []
  tc = 0 
  #print(beats_times)
  for i in range(len(array_fs)): 

    if(i<len(array_fs) -2 ): 

      while((beats_times[tc] > array_fs[i] and beats_times[tc] < array_fs[i+1]) and (tc < beats_times.size - 1 )): 
        #same values may be present
        #controllo su length di onset times !!!! 
        if((tc+1)%ts==0):
            new_beats.append(i)
          #if((beats_times[tc] > array_fs[i] + ts_fs/2) and (beats_times[tc] < array_fs[i] + ts_fs)): 
          #  new_beats.append(i)
          #elif((beats_times[tc] > array_fs[i]) and (beats_times[tc] < array_fs[i] + ts_fs/2)): 
          #  new_beats.append(i+1)
        tc = tc +1
  #print(new_beats)
    sd_list_n = ""
    beats_ar = np.full(len(array_fs), 0)

    for i in range(len(beats_ar)): 
        for b in new_beats:
            if(i == b): 
                beats_ar[i] = random.randint(-30, 30)

    for i in range(len(beats_ar)) :
        if(i != len(beats_ar) -1  ): 
            sd_list_n = sd_list_n + str(i) + ":("+ str(beats_ar[i]) +"), "
        else: 
            sd_list_n = sd_list_n + str(i) + ":("+ str(beats_ar[i]) +")"
    return sd_list_n

