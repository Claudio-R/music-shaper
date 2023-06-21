import base64
from requests import post,get
import json
import numpy as np 
import pandas as pd 

#Libraries to create the multiclass model
from keras.models import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasClassifier
from keras.utils import np_utils
#Import tensorflow and disable the v2 behavior and eager mode
import tensorflow as tf
tf.compat.v1.disable_eager_execution()
tf.compat.v1.disable_v2_behavior()

#Library to validate the model
from sklearn.model_selection import cross_val_score, KFold, train_test_split
from sklearn.preprocessing import LabelEncoder,MinMaxScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import confusion_matrix, accuracy_score

client_id ='cc1298c00eee4bf8a344bfb2056ac496'
client_secret = '27d6b03a2d8048e2bb48fe2d55d78aae'

def get_token():
    auth_string = client_id + ":" + client_secret
    auth_bytes = auth_string.encode("utf-8")
    auth_base64 = str(base64.b64encode(auth_bytes), "utf-8")

    #hhtp we want to make request to 
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

def get_auth_header(token): 
    return{"Authorization": "Bearer "+ token}


#Must match the one in the application in dashboard
#Credentials to access the Spotify Music Data
#manager = SpotifyClientCredentials(client_id,client_secret)
#sp = spotipy.Spotify(client_credentials_manager=manager)

#search for an artist, get the id, get the songs of the artists
def search_for_song(token, artist_name, song_name):
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
    return json_result["tracks"]["items"][0]["id"]



def search_for_song_test(token, artist_name, song_name):
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
    return json_result["tracks"]["items"][0]


#useless, from tutorial
def get_songs_by_artist(token, artist_id): 
    url = f"https://api.spotify.com/v1/artists/{artist_id}/top-tracks?country=US"
    headers = get_auth_header(token)
    result = get(url, headers=headers)
    json_result = json.loads(result.content)["tracks"]
    return json_result

#taking audio features given id of song
def get_audio_features_given_id(token, id): 
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

token  = get_token()
id = search_for_song(token, "White lies", "Unfinished business")
features = get_audio_features_given_id(token, id)

df = pd.read_csv("/content/drive/MyDrive/data_moods.csv")
col_features = df.columns[6:-3]
X = MinMaxScaler().fit_transform(df[col_features])
X2 = np.array(df[col_features])
Y = df["mood"]

encoder = LabelEncoder()
encoder.fit(Y)
encoded_y = encoder.transform(Y)
X_train,X_test,Y_train,Y_test = train_test_split(X,encoded_y,test_size=0.2,random_state=15)
target = pd.DataFrame({'mood':df['mood'].tolist(),'encode':encoded_y}).drop_duplicates().sort_values(['encode'],ascending=True)

def base_model():
    #Create the model
    model = Sequential()
    #Add 1 layer with 8 nodes,input of 4 dim with relu function
    model.add(Dense(8,input_dim=10,activation='relu'))
    #Add 1 layer with output 3 and softmax function
    model.add(Dense(4,activation='softmax'))
    #Compile the model using sigmoid loss function and adam optim
    model.compile(loss='categorical_crossentropy',optimizer='adam',
                 metrics=['accuracy'])
    return model

estimator = KerasClassifier(build_fn=base_model,epochs=300,batch_size=200,verbose=0)

#Evaluate the model using KFold cross validation
kfold = KFold(n_splits=10,shuffle=True)
results = cross_val_score(estimator,X,encoded_y,cv=kfold)

estimator.fit(X_train,Y_train)
y_preds = estimator.predict(X_test)
y_prob_pred = estimator.predict_proba(X_test)

def predict_mood(id_song, token):
    #Join the model and the scaler in a Pipeline
    pip = Pipeline([('minmaxscaler',MinMaxScaler()),('keras',KerasClassifier(build_fn=base_model,epochs=300,
                                                                             batch_size=200,verbose=0))])
    #Fit the Pipeline
    pip.fit(X2,encoded_y)

    #Obtain the features of the song
    preds = get_audio_features_given_id(token, id = id_song)
   
    #Pre-process the features to input the Model
    preds_features = np.array(preds).reshape(-1,1).T

    #Predict the features of the song
    results = pip.predict(preds_features)
    results_prob = pip.predict_proba(preds_features)
    #print(results_prob)
    max_prob_indx = np.where(results_prob== np.amax(results_prob))

    results_prob[0, max_prob_indx[1][0]]= 0
    
    sec_prob_indx = np.where(results_prob== np.amax(results_prob))
    results_prob[0, sec_prob_indx[1][0]]= 0
    third_prob_indx = np.where(results_prob== np.amax(results_prob))
    results_prob[0, third_prob_indx[1][0]]= 0
    fourth_prob_indx = np.where(results_prob== np.amax(results_prob))
    mood = target['mood'][target['encode']==int(max_prob_indx[1][0])]
    mood2 = target['mood'][target['encode']==int(sec_prob_indx[1][0])]
    mood3 = target['mood'][target['encode']==int(third_prob_indx[1][0])]
    mood4 = target['mood'][target['encode']==int(fourth_prob_indx[1][0])]

    moods = []
    moods.append(mood)
    moods.append(mood2)
    moods.append(mood3)
    moods.append(mood4)
    return moods
    
def moods_prediction(artist, song): 
    token = get_token()
    #Artist - song name
    
    id = search_for_song(token, artist, song)
    
    moods = []
    moods =  predict_mood(id, token)

    predicted_moods = []
    m1 = moods[0].values
    m2 = moods[1].values
    m3 = moods[2].values
    m4 = moods[3].values
    #Predicted moods in as array of 
    predicted_moods.append(m1)
    predicted_moods.append(m2) 
    predicted_moods.append(m3) 
    predicted_moods.append(m4) 

    return predicted_moods

def moods_prediction_finale(song, artist): 
    energy_calm = False
    happy_sad = False
    predicted_moods = moods_prediction(song, artist)
    final_moods = []
    for mood in predicted_moods:

        if((mood[0] == 'Energetic' or mood[0] == 'Calm') and energy_calm == False): 
            final_moods.append(mood[0])
            energy_calm = True
    
        if ( (mood[0] == 'Happy' or mood[0] == 'Sad') and happy_sad == False): 
            final_moods.append(mood[0])
            happy_sad = True

    return final_moods 
   
