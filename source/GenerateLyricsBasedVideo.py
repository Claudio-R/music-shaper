import subprocess, time, os, sys

#SECTION - Setup Environment
sub_p_res = subprocess.run(['nvidia-smi', '--query-gpu=name,memory.total,memory.free', '--format=csv,noheader'], stdout=subprocess.PIPE).stdout.decode('utf-8')

def setup_environment():
    try:
        ipy = get_ipython()
    except:
        ipy = 'could not get_ipython'

    if 'google.colab' in str(ipy):
        start_time = time.time()
        packages = [
            'triton xformers',
            'einops==0.4.1 pytorch-lightning==1.7.7 torchdiffeq==0.2.3 torchsde==0.2.5',
            'ftfy timm transformers open-clip-torch omegaconf torchmetrics',
            'safetensors kornia accelerate jsonmerge matplotlib resize-right',
            'scikit-learn numpngw pydantic'
        ]
        for package in packages:
            print(f"..installing {package}")
            subprocess.check_call([sys.executable, '-m', 'pip', 'install'] + package.split())
        if not os.path.exists("deforum-stable-diffusion"):
            subprocess.check_call(['git', 'clone', '-b', '0.7.1', 'https://github.com/deforum-art/deforum-stable-diffusion.git'])
        else:
            print(f"..deforum-stable-diffusion already exists")
        with open('deforum-stable-diffusion/src/k_diffusion/__init__.py', 'w') as f:
            f.write('')
        sys.path.extend(['deforum-stable-diffusion/','deforum-stable-diffusion/src',])
        end_time = time.time()
        print(f"..environment set up in {end_time-start_time:.0f} seconds")
    else:
        sys.path.extend(['src'])
        print("..skipping setup")

setup_environment()

import random
import clip
from IPython import display
from types import SimpleNamespace
from helpers.save_images import get_output_folder
from helpers.settings import load_args
from helpers.render import render_animation, render_input_video, render_image_batch, render_interpolation
from helpers.model_load import get_model_output_paths
from helpers.aesthetics import load_aesthetics_model
from helpers.prompts import Prompts

def PathSetup():
    models_path = "models" #@param {type:"string"}
    configs_path = "configs" #@param {type:"string"}
    output_path = "outputs" #@param {type:"string"}
    mount_google_drive = True #@param {type:"boolean"}
    models_path_gdrive = "../AI/models" #@param {type:"string"}
    output_path_gdrive = "../AI/StableDiffusion" #@param {type:"string"}
    return locals()

root = SimpleNamespace(**PathSetup())
root.models_path, root.output_path = get_model_output_paths(root)

import numpy as np
from youtube_transcript_api import YouTubeTranscriptApi
import requests
from os import path
import re;

import openai
import time
import spacy
nlp = spacy.load("en_core_web_sm")

import nltk
nltk.download('words')

from nltk.corpus import words
dictionary = set(words.words())

import MusicEmotionRecognition
import SpotifyLyrics
import Utils

def searchSongOnYt(artist, title, needLyrics = True):
    # Your YouTube Data API v3 key
    api_key = "AIzaSyBcdBPEmSRGErb2s1gfQ8OEoOBjfEaJBf0"

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

def getLyricsFromYT(yt_video_id):
    lyrics = ""
    try:
        lyrics = YouTubeTranscriptApi.get_transcript(yt_video_id, languages=['en', 'en-US'])
    except Exception as e:
        print(f"Error: {e}")

    return lyrics

def printLyrics(lyrics):
    print("Requested lyrics:")
    if type(lyrics) == str:
        print(lyrics)
    else:
        for item in lyrics:
            print(item)

def importLyricsFromFile():
    return ""

def getLyrics(artist, song):
    try:
        lyrics = Utils.search_on_spotify(artist, song)
    except:
        print("Exception Occurred in search_on_spotify function: The song has no lyrics or doesn't exist")
        lyrics=[]

    if not lyrics:
        id = searchSongOnYt(artist, song)
        lyrics = getLyricsFromYT(id)

    return lyrics


# Retrieve Artist and Song Name from the file
with open('../server/names.txt') as f:
    line = f.readline()

x = line.split("&")
artist = x[0]
title = x[1]

# Retrieve Lyrics
lyrics = getLyrics(artist, title)
printLyrics(lyrics)

# Download the song from YouTube
video_id = searchSongOnYt(artist, title, needLyrics = False)
link = 'https://www.youtube.com/watch?v=' + video_id
print("Link: ", link)

# specify the outputPath without extension
outPath = "/content/drive/MyDrive/YT_downloads/{}".format(artist)
Utils.downloadSongFromYT(link, outPath)

#!SECTION

#SECTION - Text Processing
debug_sentence_check = True #@param {type:"boolean"}
def is_meaningless_sentence(sentence):
    sentence = sentence.replace(", ", " ")
    words = re.split(r' ', sentence)
    wordsCount = 0
    nonWordsCount = 0

    for word in words:
        #if debug_sentence_check:
        #print(word)
        valid_word = word.lower() in dictionary
        #if debug_sentence_check:
        #print(valid_word)
        if valid_word:
            wordsCount += 1
        else:
            nonWordsCount += 1

    result = nonWordsCount > wordsCount
    return result


debug_text_processing = True #@param {type:"boolean"}
remove_meaningless_text = True #@param {type:"boolean"}
remove_repeated_sentences = True #@param {type:"boolean"}

fullText = ""
textTimingArrayOriginal = []
currentText = ""
for item in lyrics:
    #remove text with special character ♪
    item["text"] = re.sub("( )*♪( )*", "", item["text"])

    #remove text in parenthesis
    item["text"] = re.sub("\[(.*?)\]", "", item["text"])

    item["text"] = re.sub("\((.*?)\)", "", item["text"])

    #replace some special characters with spaces
    item["text"] = item["text"].replace("\n", " ")

    item["text"] = item["text"].replace(u'\xa0', u' ')

    #strip text from both ends
    item["text"] = item["text"].strip()

    #remove items with empty text
    if item["text"] == "":
      del item
      continue

    #remove meaningless text (if enabled)
    if remove_meaningless_text:
      if is_meaningless_sentence(item["text"]):
        #print("!!!sentence is meaningless and will be removed from the lyrics: " + item["text"])
        del item
        continue

    #remove repeated sentences (if enabled)
    if remove_repeated_sentences:
      if item["text"] == currentText:
        del item
        continue

    currentText = item["text"]
    fullText = fullText + item["text"] + " ";
    textTimingArrayOriginal.append([str(item["start"]), item["text"]]);

fullText = fullText[:-1]

#ANCHOR - Specify Intervals
specify_time_interval = True #@param {type:"boolean"}
start_time_sec = 51 #@param {type:"number"}
end_time_sec = 76 #@param {type:"number"}

textTimingArray =[]
maxLength = len(textTimingArrayOriginal)
if specify_time_interval:
    start_index = 0
    curItem = textTimingArrayOriginal[start_index]

    while start_index < maxLength and float(curItem[0]) < start_time_sec:
        start_index += 1
        curItem = textTimingArrayOriginal[start_index]
    end_index = maxLength - 1
    curItem = textTimingArrayOriginal[end_index]

    while end_index >=0 and float(curItem[0]) > end_time_sec:
        end_index -= 1
        curItem = textTimingArrayOriginal[end_index]

    assert start_index <= end_index, f"Error: start index > end index! Check the algorithm"

    print("End index: ", end_index, "item: ", textTimingArrayOriginal[end_index])
    textTimingArray = textTimingArrayOriginal[start_index:end_index+1]

else:
    textTimingArray = textTimingArrayOriginal

if specify_time_interval:
    Utils.cutAudio(outPath + ".wav", outPath + "_cut.wav", start_time_sec, end_time_sec)

#ANCHOR - Split text into sentences
min_words_count_in_sentence = 10 #@param {type:"number"}

def getWordsCount(text):
    # Process the text
    doc = nlp(text)
    count = 0
    # Iterate over each token in the processed text
    for token in doc:
        if not token.is_punct:
            count+=1
    return count

minCount = min_words_count_in_sentence

def getSentences(textTimingArray):
    sentence_array = []
    timing_array = []
    curText = ""
    curTiming = 0
    for index, item in enumerate(textTimingArray):
        if not curText:
            curText = item[1]
            curTiming = item[0]
        else:
            curText = curText + ", " + item[1]

    wordsCount = getWordsCount(curText)
    if wordsCount >= minCount:
        sentence_array.append(curText)
        timing_array.append(curTiming)
        curText = ""
    elif index == len(textTimingArray)-1:
        sentence_array.append(curText)
        timing_array.append(curTiming)

    return (sentence_array, timing_array)

(sentence_array, timing_array) = getSentences(textTimingArray)
#!SECTION - Text Processing

#SECTION - Define the style
#ANCHOR - Mood Detection
moods = MusicEmotionRecognition.moods_prediction_finale(artist, title)
print('Moods: ', moods)

#ANCHOR - Onset Detection
fps = 10
tempo,ts = Utils.get_tempo_ts(artist, title)
zoom_librosa = Utils.get_beats_librosa_zoom(outPath + "_cut.wav", fps, tempo, ts)
angles_librosa = Utils.get_beats_librosa_angle(outPath + "_cut.wav", fps, tempo, ts )

#ANCHOR - Retrieve styles from the database
with open('../server/style.txt') as f:
    line = f.readline()
y = line.split("&")
print(y)

style_1 = y[0]
style_2 = y[1]
content_type = y[2]

# content_type = 'Digital paint' #@param ['Image', 'Photograph', 'Painting', 'Sketch', '3D render', 'Digital paint', 'Composition', 'Analog photo'] {type:'string'}
# style_1 = 'highly detailed'#@param ["realistic" ,"oilpainting", "modern", "cyberpunk", "futuristic", "Surrealist" ,  "trending on artstation",  "matte",  "elegant",  "illustration" , "epic", "beautiful", "dreamy", "colorful", "watercolors", "highly detailed", "grainy", "psychedelic"]
# style_2 = 'beautiful' #@param ["realistic" ,"oilpainting", "modern", "cyberpunk", "futuristic", "Surrealist" ,  "trending on artstation",  "matte",  "elegant",  "illustration" , "epic", "beautiful", "dreamy", "colorful", "watercolors", "highly detailed", "grainy", "psychedelic"]
#!SECTION - Define the style


#SECTION - Generate prompts
#ANCHOR - Generate Input prompts
debug_generation_prompts = True #@param {type:"boolean"}

openai_api_key = "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" #@param {type:"string"}
openai.api_key = openai_api_key

def chat_with_chatgpt(prompt):
  temperature = 1.0
  max_tokens = 300

  # Generate a response using the OpenAI API
  response = openai.Completion.create(
      engine="text-davinci-002",
      prompt=prompt,
      temperature=temperature,
      max_tokens=max_tokens,
  )
  answer = response.choices[0].text.strip()
  return answer

def generate_prompts(sentence_array, content, style1, style2):
    c = content
    s1 = style1
    s2 = style2

    if debug_generation_prompts:
        print("Desired content: ", c)
        print("Desired 1st style: ", s1)
        print("Desired 2nd style: ", s2)

    chatgpt_prompts = []

    for sentence in sentence_array:
        if debug_generation_prompts:
            print("\nCurrent sentence: ", sentence)

        #prompt = "Generate a prompt for a text2image model based on stable diffusion, \
        #Describing the phrase '" + itm + "' as a " + c +", with style "+ s1 +", " + s2 +".Respecting \
        #the following order, provide a single output that MUST include (Some examples are provided, \
        #but the one written at the beginning must be included) : Content type (photograph, drawing, sketch, \
        #3D render, but INCLUDE THE ONE DEFINED AT THE BEGINNING),  Description (what's happening in the scene, \
        #in third person, cleary defining the Content as subject, not impersonal form), Art Style and details \
        #definition (realistic, oilpainting, modern, 3D)."
        #print("Prompt:"§

        '''prompt = "Generate a CONCISE prompt for a text2image model based on stable diffusion.\
                With "+ s1 +", " + s2 +" style. The prompt should describe the phrase '" + sentence + "' as a " + c +"\
                .Respecting the following order, provide a single output phrase that MUST ALWAYS include\
                : Content type, Description (what's happening in the scene, in third person, cleary defining the Content as subject, \
                not impersonal form), Art Style and details definition. Respect the form : 'Content of Description in these Styles'.\
                Don't mention stable diffusion model, but just a prompt that would work with it. Rephrase the input, DON'T repeat it"
        '''
        prompt = "Generate a CONCISE prompt for a text2image model based on stable diffusion.\
                With "+ s1 +", " + s2 +" style, expressing a "+ moods[0] +" and "+ moods[1] +" feeling . The prompt should describe the phrase '" + sentence + "' as a " + c +"\
                .Respecting the following order, provide a single output phrase that MUST ALWAYS include\
                : Content type, Description (what's happening in the scene, in third person, cleary defining the Content as subject, \
                not impersonal form), Art Style and details definition. Respect the form : 'Content of Description in these Styles'.\
                Don't mention stable diffusion model, but just a prompt that would work with it. Rephrase the input, DON'T repeat it"

        response = chat_with_chatgpt(prompt)

        if debug_generation_prompts:
            print("\nGenerated prompt: ", response)
            print("----")

        chatgpt_prompts.append(response)

        #because of openAI API limitations
        time.sleep(1)
    
    return chatgpt_prompts

chatgpt_prompts = generate_prompts(sentence_array, content_type, style_1, style_2)

#ANCHOR - Generate Animation Prompts
debug_animation_prompts = True #@param {type:"boolean"}

fps=10 #@param {type:"number"}
frames_array = []

firstTiming = float(timing_array[0])
for i in range(len(chatgpt_prompts)):
    #get frames array
    curTiming = float(timing_array[i])
    if specify_time_interval:
        curTiming -= start_time_sec

    frames_array.append(int(curTiming*fps))
    #do some minor text processing with prompts from chatGPT
    chatgpt_prompts[i] = chatgpt_prompts[i].replace("\n", " ")
    chatgpt_prompts[i] = chatgpt_prompts[i].replace(".", " ")
    chatgpt_prompts[i] = chatgpt_prompts[i].replace("!", " ")
    chatgpt_prompts[i] = chatgpt_prompts[i].replace(":", " ")
    chatgpt_prompts[i] = chatgpt_prompts[i].strip()

animation_prompts = dict(zip(frames_array, chatgpt_prompts))

if debug_animation_prompts:
    print("Frames: \n", frames_array)
    print("Animation prompts: \n", animation_prompts)

neg_prompts = {}

#!SECTION - Generate prompts

#SECTION - Generate clip
#ANCHOR - Settings for Deforum Stable Diffusion
def DeforumAnimArgs():
    #@markdown ####**Animation:**
    animation_mode = '2D' #@param ['None', '2D', '3D', 'Video Input', 'Interpolation'] {type:'string'}
    #finish this
    max_frames = frames_array[-1] + 100
    border = 'wrap' #@param ['wrap', 'replicate'] {type:'string'}
    if(specify_time_interval):
        max_frames = int((end_time_sec - start_time_sec)*fps)
    #@markdown ####**Motion Parameters:**
    angle = angles_librosa#@param {type:"string"}
    zoom = zoom_librosa#@param {type:"string"}
    translation_x = "0:(2)"#@param {type:"string"}
    translation_y = "0:(2)"#@param {type:"string"}
    translation_z = "0:(0)"#@param {type:"string"}
    rotation_3d_x = "0:(0)"#@param {type:"string"}
    rotation_3d_y = "0:(0)"#@param {type:"string"}
    rotation_3d_z = "0:(0)"#@param {type:"string"}

    flip_2d_perspective = False #{type:"boolean"}
    perspective_flip_theta = "0:(0)"# {type:"string"}
    perspective_flip_phi = "0:(t%15)"#{type:"string"}
    perspective_flip_gamma = "0:(0)"# {type:"string"}
    perspective_flip_fv = "0:(53)"# {type:"string"}
    noise_schedule = "0: (0.05)"#{type:"string"}
    strength_schedule = "0: (0.65)"#{type:"string"}
    contrast_schedule = "0: (1.0)"#{type:"string"}
    hybrid_comp_alpha_schedule = "0:(1)" # {type:"string"}
    hybrid_comp_mask_blend_alpha_schedule = "0:(0.5)" # {type:"string"}
    hybrid_comp_mask_contrast_schedule = "0:(1)" # {type:"string"}
    hybrid_comp_mask_auto_contrast_cutoff_high_schedule =  "0:(100)" # {type:"string"}
    hybrid_comp_mask_auto_contrast_cutoff_low_schedule =  "0:(0)" # {type:"string"}

    #Sampler Scheduling:
    enable_schedule_samplers = False
    sampler_schedule = "0:('euler'),10:('dpm2'),20:('dpm2_ancestral'),30:('heun'),40:('euler'),50:('euler_ancestral'),60:('dpm_fast'),70:('dpm_adaptive'),80:('dpmpp_2s_a'),90:('dpmpp_2m')" #@param {type:"string"}

    #@markdown ####**Unsharp mask (anti-blur) Parameters:**
    kernel_schedule = "0: (5)"#{type:"string"}
    sigma_schedule = "0: (1.0)"#{type:"string"}
    amount_schedule = "0: (0.2)"#{type:"string"}
    threshold_schedule = "0: (0.0)"#{type:"string"}

    #@markdown ####**Coherence:**
    color_coherence = 'Match Frame 0 LAB' #@param ['None', 'Match Frame 0 HSV', 'Match Frame 0 LAB', 'Match Frame 0 RGB', 'Video Input'] {type:'string'}
    color_coherence_video_every_N_frames = 1 #@param {type:"integer"}
    color_force_grayscale = False #@param {type:"boolean"}
    diffusion_cadence = '1' #@param ['1','2','3','4','5','6','7','8'] {type:'string'}

    #@markdown ####**3D Depth Warping:**
    use_depth_warping = True #@param {type:"boolean"}
    midas_weight = 0.3#@param {type:"number"}
    near_plane = 200
    far_plane = 10000
    fov = 40#@param {type:"number"}
    padding_mode = 'border'#@param ['border', 'reflection', 'zeros'] {type:'string'}
    sampling_mode = 'bicubic'#@param ['bicubic', 'bilinear', 'nearest'] {type:'string'}
    save_depth_maps = False #@param {type:"boolean"}

    #@markdown ####**Video Input:**
    #video_init_path ='/content/video_in.mp4'#@param {type:"string"}
    video_init_path =''#@param {type:"string"}
    extract_nth_frame = 1#@param {type:"number"}
    overwrite_extracted_frames = True #@param {type:"boolean"}
    use_mask_video = False #@param {type:"boolean"}
    #video_mask_path ='/content/video_in.mp4'#@param {type:"string"}
    video_mask_path =''#@param {type:"string"}

    #@markdown ####**Hybrid Video for 2D/3D Animation Mode:**
    hybrid_generate_inputframes = False # {type:"boolean"}
    hybrid_use_first_frame_as_init_image = True # {type:"boolean"}
    hybrid_motion = "None" # ['None','Optical Flow','Perspective','Affine']
    hybrid_motion_use_prev_img = False # {type:"boolean"}
    hybrid_flow_method = "DIS Medium" # ['DenseRLOF','DIS Medium','Farneback','SF']
    hybrid_composite = False # {type:"boolean"}
    hybrid_comp_mask_type = "None" # ['None', 'Depth', 'Video Depth', 'Blend', 'Difference']
    hybrid_comp_mask_inverse = False # {type:"boolean"}
    hybrid_comp_mask_equalize = "None" #  ['None','Before','After','Both']
    hybrid_comp_mask_auto_contrast = False # {type:"boolean"}
    hybrid_comp_save_extra_frames = False # {type:"boolean"}
    hybrid_use_video_as_mse_image = False # {type:"boolean"}


    #@markdown ####**Interpolation:**
    interpolate_key_frames = True #@param {type:"boolean"}
    interpolate_x_frames = 20 #@param {type:"number"}

    #@markdown ####**Resume Animation:**
    resume_from_timestring = False #@param {type:"boolean"}
    resume_timestring = "20220829210106" #@param {type:"string"}

    return locals()

override_settings_with_file = False # {type:"boolean"}
settings_file = "custom" # ["custom", "512x512_aesthetic_0.json","512x512_aesthetic_1.json","512x512_colormatch_0.json","512x512_colormatch_1.json","512x512_colormatch_2.json","512x512_colormatch_3.json"]
custom_settings_file = "/content/drive/MyDrive/Settings.txt"# {type:"string"}

def DeforumArgs():
    #@markdown **Image Settings**
    W = 512 #@param
    H = 512 #@param
    W, H = map(lambda x: x - x % 64, (W, H))  # resize to integer multiple of 64
    bit_depth_output = 8 #@param [8, 16, 32] {type:"raw"}

    #@markdown **Sampling Settings**
    seed = -1 #@param
    sampler = 'euler_ancestral' #@param ["klms","dpm2","dpm2_ancestral","heun","euler","euler_ancestral","plms", "ddim", "dpm_fast", "dpm_adaptive", "dpmpp_2s_a", "dpmpp_2m"]
    steps = 15 #@param
    scale = 7 #@param
    ddim_eta = 0.0 #@param
    dynamic_threshold = None
    static_threshold = None

    #@markdown **Save & Display Settings**
    save_samples = True #@param {type:"boolean"}
    save_settings = True #@param {type:"boolean"}
    display_samples = True #@param {type:"boolean"}
    save_sample_per_step = False #@param {type:"boolean"}
    show_sample_per_step = False #@param {type:"boolean"}

    #@markdown **Batch Settings**
    n_batch = 1 #@param
    n_samples = 1 #@param
    batch_name = "kate" #@param {type:"string"}
    filename_format = "{timestring}_{index}_{prompt}.png" #@param ["{timestring}_{index}_{seed}.png","{timestring}_{index}_{prompt}.png"]
    seed_behavior = "iter" #@param ["iter","fixed","random","ladder","alternate"]
    seed_iter_N = 1 #@param {type:'integer'}
    make_grid = False #@param {type:"boolean"}
    grid_rows = 2 #@param
    outdir = get_output_folder(root.output_path, batch_name)

    #@markdown **Init Settings**
    use_init = False #{type:"boolean"}
    strength = 0.65 #{type:"number"}
    strength_0_no_init = True # Set the strength to 0 automatically when no init image is used
    init_image = "https://cdn.pixabay.com/photo/2022/07/30/13/10/green-longhorn-beetle-7353749_1280.jpg" #@param {type:"string"}
    add_init_noise = False #{type:"boolean"}
    init_noise = 0.01 #
    # Whiter areas of the mask are areas that change more
    use_mask = False # {type:"boolean"}
    use_alpha_as_mask = False # use the alpha channel of the init image as the mask
    mask_file = "https://www.filterforge.com/wiki/images/archive/b/b7/20080927223728%21Polygonal_gradient_thumb.jpg" #@param {type:"string"}
    invert_mask = False # {type:"boolean"}
    # Adjust mask image, 1.0 is no adjustment. Should be positive numbers.
    mask_brightness_adjust = 1.0  # {type:"number"}
    mask_contrast_adjust = 1.0  # {type:"number"}
    # Overlay the masked image at the end of the generation so it does not get degraded by encoding and decoding
    overlay_mask = True  # {type:"boolean"}
    # Blur edges of final overlay mask, if used. Minimum = 0 (no blur)
    mask_overlay_blur = 5 # {type:"number"}

    #@markdown **Exposure/Contrast Conditional Settings**
    mean_scale = 0 # {type:"number"}
    var_scale = 0 # {type:"number"}
    exposure_scale = 0 # {type:"number"}
    exposure_target = 0.5 # {type:"number"}

    #@markdown **Color Match Conditional Settings**
    colormatch_scale = 0 # {type:"number"}
    colormatch_image = "https://www.saasdesign.io/wp-content/uploads/2021/02/palette-3-min-980x588.png" # {type:"string"}
    colormatch_n_colors = 4 # {type:"number"}
    ignore_sat_weight = 0 #{type:"number"}

    #@markdown **CLIP\Aesthetics Conditional Settings**
    clip_name = 'ViT-L/14' #['ViT-L/14', 'ViT-L/14@336px', 'ViT-B/16', 'ViT-B/32']
    clip_scale = 0 # {type:"number"}
    aesthetics_scale = 0 # {type:"number"}
    cutn = 1 # {type:"number"}
    cut_pow = 0.0001 # {type:"number"}

    #@markdown **Other Conditional Settings**
    init_mse_scale = 0 # {type:"number"}
    init_mse_image = "https://cdn.pixabay.com/photo/2022/07/30/13/10/green-longhorn-beetle-7353749_1280.jpg" # {type:"string"}
    blue_scale = 0 #@param {type:"number"}

    #@markdown **Conditional Gradient Settings**
    gradient_wrt = 'x0_pred' #@param ["x", "x0_pred"]
    gradient_add_to = 'both' #@param ["cond", "uncond", "both"]
    decode_method = 'linear' #@param ["autoencoder","linear"]
    grad_threshold_type = 'dynamic' #@param ["dynamic", "static", "mean", "schedule"]
    clamp_grad_threshold = 0.2 #@param {type:"number"}
    clamp_start = 0.2 #@param
    clamp_stop = 0.01 #@param
    grad_inject_timing = list(range(1,10)) #@param

    #@markdown **Speed vs VRAM Settings**
    cond_uncond_sync = True #@param {type:"boolean"}
    precision = 'autocast'
    C = 4
    f = 8

    cond_prompt = ""
    cond_prompts = ""
    uncond_prompt = ""
    uncond_prompts = ""
    timestring = ""
    init_latent = None
    init_sample = None
    init_sample_raw = None
    mask_sample = None
    init_c = None
    seed_internal = 0
    return locals()


args_dict = DeforumArgs()
anim_args_dict = DeforumAnimArgs()

if override_settings_with_file:
    load_args(args_dict, anim_args_dict, settings_file, custom_settings_file, verbose=False)

args = SimpleNamespace(**args_dict)
anim_args = SimpleNamespace(**anim_args_dict)

args.timestring = time.strftime('%Y%m%d%H%M%S')
args.strength = max(0.0, min(1.0, args.strength))

# Load clip model if using clip guidance
if (args.clip_scale > 0) or (args.aesthetics_scale > 0):
    root.clip_model = clip.load(args.clip_name, jit=False)[0].eval().requires_grad_(False).to(root.device)
    if (args.aesthetics_scale > 0):
        root.aesthetics_model = load_aesthetics_model(args, root)

if args.seed == -1:
    args.seed = random.randint(0, 2**32 - 1)
if not args.use_init:
    args.init_image = None
if args.sampler == 'plms' and (args.use_init or anim_args.animation_mode != 'None'):
    print(f"Init images aren't supported with PLMS yet, switching to KLMS")
    args.sampler = 'klms'
if args.sampler != 'ddim':
    args.ddim_eta = 0

if anim_args.animation_mode == 'None':
    anim_args.max_frames = 1
elif anim_args.animation_mode == 'Video Input':
    args.use_init = True

#ANCHOR - Frames Generation
# get prompts
cond, uncond = Prompts(prompt=animation_prompts).as_dict()

# dispatch to appropriate renderer
if anim_args.animation_mode == '2D' or anim_args.animation_mode == '3D':
    render_animation(root, anim_args, args, cond, uncond)
elif anim_args.animation_mode == 'Video Input':
    render_input_video(root, anim_args, args, cond, uncond)
elif anim_args.animation_mode == 'Interpolation':
    render_interpolation(root, anim_args, args, cond, uncond)
else:
    render_image_batch(root, args, cond, uncond)

#ANCHOR - Video Generation
skip_video_for_run_all = False #@param {type: 'boolean'}
fps = 10 #@param {type:"number"}
#@markdown **Manual Settings**
use_manual_settings = False #@param {type:"boolean"}
image_path = "" #@param {type:"string"}
mp4_path = "" #@param {type:"string"}
render_steps = False  #@param {type: 'boolean'}
path_name_modifier = "x0_pred" #@param ["x0_pred","x"]
make_gif = False
bitdepth_extension = "exr" if args.bit_depth_output == 32 else "png"

if skip_video_for_run_all == True:
    print('Skipping video creation, uncheck skip_video_for_run_all if you want to run it')
else:
    import os
    import subprocess
    from base64 import b64encode

    print(f"{image_path} -> {mp4_path}")

    if use_manual_settings:
        max_frames = "200" #@param {type:"string"}
    else:
        if render_steps: # render steps from a single image
            fname = f"{path_name_modifier}_%05d.png"
            all_step_dirs = [os.path.join(args.outdir, d) for d in os.listdir(args.outdir) if os.path.isdir(os.path.join(args.outdir,d))]
            newest_dir = max(all_step_dirs, key=os.path.getmtime)
            image_path = os.path.join(newest_dir, fname)
            print(f"Reading images from {image_path}")
            mp4_path = os.path.join(newest_dir, f"{args.timestring}_{path_name_modifier}.mp4")
            max_frames = str(args.steps)
        else: # render images for a video
            image_path = os.path.join(args.outdir, f"{args.timestring}_%05d.{bitdepth_extension}")
            mp4_path = os.path.join(args.outdir, f"{args.timestring}.mp4")
            print("image path: ", image_path)
            print("mp4_path: ", mp4_path)
            max_frames = str(anim_args.max_frames)

    # make video
    cmd = [
        'ffmpeg',
        '-y',
        '-vcodec', bitdepth_extension,
        '-r', str(fps),
        '-start_number', str(0),
        #'-i', audio_path,
        '-i', image_path,
        '-frames:v', max_frames,
        '-c:v', 'libx264',
        '-vf',
        f'fps={fps}',
        '-pix_fmt', 'yuv420p',
        '-crf', '17',
        '-preset', 'veryfast',
        '-pattern_type', 'sequence',
        mp4_path
    ]
    process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    stdout, stderr = process.communicate()
    if process.returncode != 0:
        print(stderr)
        raise RuntimeError(stderr)

    mp4 = open(mp4_path,'rb').read()
    data_url = "data:video/mp4;base64," + b64encode(mp4).decode()
    display.display(display.HTML(f'<video controls loop><source src="{data_url}" type="video/mp4"></video>') )

    if make_gif:
        gif_path = os.path.splitext(mp4_path)[0]+'.gif'
        cmd_gif = [
            'ffmpeg',
            '-y',
            '-i', mp4_path,
            '-r', str(fps),
            gif_path
        ]
        process_gif = subprocess.Popen(cmd_gif, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

#ANCHOR - Add Audio
#against some strange error we sometimes get:
#https://github.com/googlecolab/colabtools/issues/3409

import locale
import os
locale.getpreferredencoding = lambda: "UTF-8"
audio_path = outPath + "_cut.wav" #"/content/drive/MyDrive/AI/Audio /Eminem_cut.mp3" #@param {type:"string"}
mp4_final_path = "/content/drive/MyDrive/AI/Video/Music_cut.mp4" #@param {type:"string"}

# run the command : !ffmpeg -i "$mp4_path" -i "$audio_path" -c:a aac -b:a 128k -map 0:v:0 -map 1:a:0 -shortest "$mp4_final_path"

cmd = [
    'ffmpeg',
    '-y',
    '-i', mp4_path,
    '-i', audio_path,
    '-c:a', 'aac',
    '-b:a', '128k',
    '-map', '0:v:0',
    '-map', '1:a:0',
    '-shortest',
    mp4_final_path
]
process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
stdout, stderr = process.communicate()
if process.returncode != 0:
    print(stderr)
    raise RuntimeError(stderr)

#!SECTION - End of the script