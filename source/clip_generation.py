#SECTION - Setup Environment

import subprocess, time, os, sys
import re, yaml, locale
import random
from IPython import display
from types import SimpleNamespace

sub_p_res = subprocess.run(['nvidia-smi', '--query-gpu=name,memory.total,memory.free', '--format=csv,noheader'], stdout=subprocess.PIPE).stdout.decode('utf-8')

def setup_environment():
    start_time = time.time()
    packages = [
        'triton xformers',
        'einops==0.4.1 pytorch-lightning==1.7.7 torchdiffeq==0.2.3 torchsde==0.2.5',
        'ftfy timm transformers open-clip-torch omegaconf torchmetrics',
        'safetensors kornia accelerate jsonmerge matplotlib resize-right',
        'scikit-learn numpngw pydantic',
        'youtube-transcript-api pandas openai PyDictionary',
        'spacy nltk requests',
        'librosa syrics numpy seaborn',
        'keras tensorflow sklearn',
        'flask_ngrok pyngrok clip'
    ]
    for package in packages:
        print(f"..installing {package}")
        subprocess.check_call([sys.executable, '-m', 'pip', 'install'] + package.split())
    subprocess.check_call([sys.executable, '-m', 'pip', 'install', '--force-reinstall', 'https://github.com/yt-dlp/yt-dlp/archive/master.tar.gz'])

    if not os.path.exists("deforum-stable-diffusion"):
        subprocess.check_call(['git', 'clone', '-b', '0.7.1', 'https://github.com/deforum-art/deforum-stable-diffusion.git'])
    else:
        print(f"..deforum-stable-diffusion already exists")
    with open('deforum-stable-diffusion/src/k_diffusion/__init__.py', 'w') as f:
        f.write('')
    sys.path.extend(['deforum-stable-diffusion/','deforum-stable-diffusion/src',])
    end_time = time.time()
    print(f"..environment set up in {end_time-start_time:.0f} seconds")

from IPython import display
from types import SimpleNamespace

def PathSetup():
    models_path = "models" #@param {type:"string"}
    configs_path = "configs" #@param {type:"string"}
    output_path = "outputs" #@param {type:"string"}
    mount_google_drive = True #@param {type:"boolean"}
    models_path_gdrive = "/AI/models" #@param {type:"string"}
    output_path_gdrive = "/AI/StableDiffusion" #@param {type:"string"}
    return locals()
    
setup_environment()

from helpers.save_images import get_output_folder
from helpers.settings import load_args
from helpers.render import render_animation, render_input_video, render_image_batch, render_interpolation
from helpers.model_load import get_model_output_paths
from helpers.aesthetics import load_aesthetics_model
from helpers.prompts import Prompts

root = SimpleNamespace(**PathSetup())
root.models_path, root.output_path = get_model_output_paths(root)
locale.getpreferredencoding = lambda: "UTF-8"


import openai, clip
import time
import spacy
nlp = spacy.load("en_core_web_sm")

import nltk
nltk.download('words')

from nltk.corpus import words
dictionary = set(words.words())

import source.mood_prediction as mood_prediction
import source.utils as utils
import source.youtube_api as youtube_api

artist = ""
title = ""
outPath = ""
image_path = ""
mp4_path = ""
audio_path = outPath + "_cut.wav" 
mp4_final_path = "./AI/Video/Music_cut.mp4"

#!SECTION - Setup Environment
#SECTION - Settings

#ANCHOR - Text Processing
debug_sentence_check = True
debug_text_processing = True #@param {type:"boolean"}
remove_meaningless_text = True #@param {type:"boolean"}
remove_repeated_sentences = True #@param {type:"boolean"}
specify_time_interval = True #@param {type:"boolean"}
start_time_sec = 51 #@param {type:"number"}
end_time_sec = 76 #@param {type:"number"}
min_words_count_in_sentence = 10 #@param {type:"number"}

#ANCHOR - Prompts
debug_generation_prompts = True #@param {type:"boolean"}
debug_animation_prompts = True #@param {type:"boolean"}
override_settings_with_file = False # {type:"boolean"}
settings_file = "custom" # ["custom", "512x512_aesthetic_0.json","512x512_aesthetic_1.json","512x512_colormatch_0.json","512x512_colormatch_1.json","512x512_colormatch_2.json","512x512_colormatch_3.json"]
custom_settings_file = "settings.txt"# {type:"string"}

#ANCHOR - Clip
skip_video_for_run_all = False #@param {type: 'boolean'}
use_manual_settings = False #@param {type:"boolean"}
render_steps = False  #@param {type: 'boolean'}
make_gif = False
fps = 10 #@param {type:"number"}
path_name_modifier = "x0_pred" #@param ["x0_pred","x"]

#!SECTION - Settings
#SECTION - Functions

def get_lyrics():
    # Read from database
    with open('./database/names.txt') as f:
        line = f.readline()
    x = line.split("&")
    artist, title = x[0], x[1]

    # Get lyrics from Spotify or Youtube
    lyrics = utils.get_lyrics(artist, title)
    utils.print_lyrics(lyrics)

    # Download the song from YouTube
    video_id = youtube_api.search_song_on_yt(artist, title, needLyrics = False)
    link = 'https://www.youtube.com/watch?v=' + video_id
    outPath = "./database/YT_downloads/{}".format(artist)
    utils.downloadSongFromYT(link, outPath)
    return lyrics

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

def format_lyrics(lyrics):
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
    return fullText, textTimingArrayOriginal

def specify_intervals(textTimingArrayOriginal):
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
        outPath = "./database/YT_downloads/{}".format(artist)
        utils.cutAudio(outPath + ".wav", outPath + "_cut.wav", start_time_sec, end_time_sec)

    return textTimingArray

def getWordsCount(text):
    # Process the text
    doc = nlp(text)
    count = 0
    # Iterate over each token in the processed text
    for token in doc:
        if not token.is_punct:
            count+=1
    return count

def split_lyrics_into_sentences(textTimingArray):
    minCount = min_words_count_in_sentence

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

def process_text():
    '''
    Only function to be called from outside to process the text
    '''
    lyrics = get_lyrics()
    _, textTimingArrayOriginal = format_lyrics(lyrics)
    textTimingArray = specify_intervals(textTimingArrayOriginal)
    sentence_array, timing_array = split_lyrics_into_sentences(textTimingArray)
    return sentence_array, timing_array

#ANCHOR - Generate prompts

def get_moods():
    moods = mood_prediction.predict(artist, title)
    return moods

def get_zoom_angle():
    fps = 10
    tempo,ts = utils.get_tempo_ts(artist, title)
    zoom_librosa = utils.get_beats_librosa_zoom(outPath + "_cut.wav", fps, tempo, ts)
    angles_librosa = utils.get_beats_librosa_angle(outPath + "_cut.wav", fps, tempo, ts )
    return zoom_librosa, angles_librosa

def get_style():
    with open('./database/style.txt') as f:
        line = f.readline()
    y = line.split("&")

    style_1, style_2 = y[0], y[1]
    content_type = y[2]
    return style_1, style_2, content_type

def chat_with_chatgpt(prompt):
    with open("env.local.yml", 'r') as stream:
        try:
            credentials = yaml.safe_load(stream)
            openai.api_key = credentials['OPENAI_API_KEY']
        except yaml.YAMLError as exc:
            print(exc)

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

def generate_prompts(sentence_array):

    moods = get_moods()
    s1, s2, c = get_style()

    if debug_generation_prompts:
        print("Desired content: ", c)
        print("Desired 1st style: ", s1)
        print("Desired 2nd style: ", s2)

    chatgpt_prompts = []

    for sentence in sentence_array:
        if debug_generation_prompts:
            print("\nCurrent sentence: ", sentence)

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

def get_animation_prompts(sentence_array, timing_array):

    chatgpt_prompts = generate_prompts(sentence_array)
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

    return animation_prompts, neg_prompts, frames_array

def DeforumAnimArgs(frames_array):
    #@markdown ####**Animation:**
    animation_mode = '2D' #@param ['None', '2D', '3D', 'Video Input', 'Interpolation'] {type:'string'}
    #finish this
    max_frames = frames_array[-1] + 100
    border = 'wrap' #@param ['wrap', 'replicate'] {type:'string'}
    if(specify_time_interval):
        max_frames = int((end_time_sec - start_time_sec)*fps)
    #@markdown ####**Motion Parameters:**
    zoom, angles = get_zoom_angle()
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

def process_args():

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

    return args, anim_args

#ANCHOR - Generate Clip

def generate_frames(animation_prompts):
    cond, uncond = Prompts(prompt=animation_prompts).as_dict()
    args, anim_args = process_args()

    if anim_args.animation_mode == '2D' or anim_args.animation_mode == '3D':
        render_animation(root, anim_args, args, cond, uncond)
    elif anim_args.animation_mode == 'Video Input':
        render_input_video(root, anim_args, args, cond, uncond)
    elif anim_args.animation_mode == 'Interpolation':
        render_interpolation(root, anim_args, args, cond, uncond)
    else:
        render_image_batch(root, args, cond, uncond)

    if skip_video_for_run_all == True:
        print("Skipping video generation")
    else:
        import os
        import subprocess
        from base64 import b64encode

        if use_manual_settings:
            max_frames = "200" #@param {type:"string"}
        else:
            if render_steps: # render steps from a single image
                fname = f"{path_name_modifier}_%05d.png"
                all_step_dirs = [os.path.join(args.outdir, d) for d in os.listdir(args.outdir) if os.path.isdir(os.path.join(args.outdir,d))]
                newest_dir = max(all_step_dirs, key=os.path.getmtime)
                image_path = os.path.join(newest_dir, fname)
                mp4_path = os.path.join(newest_dir, f"{args.timestring}_{path_name_modifier}.mp4")
                max_frames = str(args.steps)
            else: # render images for a video
                bitdepth_extension = "exr" if args.bit_depth_output == 32 else "png"
                image_path = os.path.join(args.outdir, f"{args.timestring}_%05d.{bitdepth_extension}")
                mp4_path = os.path.join(args.outdir, f"{args.timestring}.mp4")
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
        _, stderr = process.communicate()
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
            _, stderr = process_gif.communicate()
            if process_gif.returncode != 0:
                print(stderr)
                raise RuntimeError(stderr)
            
def add_audio():
    # against some strange error we sometimes get:
    # https://github.com/googlecolab/colabtools/issues/3409

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
    
#!SECTION - Functions

#ANCHOR - Main
def generate_clip():
    sentence_array, timing_array = process_text()
    animation_prompts = get_animation_prompts(sentence_array, timing_array)
    generate_frames(animation_prompts)
    add_audio()