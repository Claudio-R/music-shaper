#Script to obtain data 
import numpy as np 
import pandas as pd 

# Libraries to create the multiclass model
from keras.models import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasClassifier

# Library to validate the model
from sklearn.preprocessing import LabelEncoder,MinMaxScaler
from sklearn.pipeline import Pipeline

#retrieve auth token - return a base 64 object
from source.spotify_api import get_spotify_token, search_for_song_id, get_audio_features_given_song_id

def base_model():
    model = Sequential(
        [
            Dense(8,input_dim=10,activation='relu'),
            Dense(4,activation='softmax')
        ]
    )
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model

def define_env():
    df = pd.read_csv("data_moods.csv")
    df.head()

    col_features = df.columns[6:-3]
    X = np.array(df[col_features])
    Y = df["mood"]

    encoder = LabelEncoder()
    encoder.fit(Y)
    encoded_y = encoder.transform(Y)

    pipe = Pipeline([
        ('minmaxscaler',MinMaxScaler()),
        ('keras', KerasClassifier(build_fn=base_model, epochs=300, batch_size=200, verbose=0)),
        ])

    pipe.fit(X, encoded_y)
    target = pd.DataFrame({'mood':df['mood'].tolist(),'encode':encoded_y}).drop_duplicates().sort_values(['encode'],ascending=True)
    # print("Labels:\n", target['mood'].values) # ['Calm', 'Energetic', 'Happy', 'Sad']

    return pipe, target

def predict(artist, song):
    song_id = search_for_song_id(artist, song)
    pipe, target = define_env()

    features = get_audio_features_given_song_id(song_id)
    features = np.array(features).reshape(-1,1).T
    results_prob = pipe.predict_proba(features)

    max_prob_indx = np.where(results_prob== np.amax(results_prob))
    results_prob[0, max_prob_indx[1][0]]= 0
    sec_prob_indx = np.where(results_prob== np.amax(results_prob))
    results_prob[0, sec_prob_indx[1][0]]= 0
    third_prob_indx = np.where(results_prob== np.amax(results_prob))
    results_prob[0, third_prob_indx[1][0]]= 0
    fourth_prob_indx = np.where(results_prob== np.amax(results_prob))
    
    mood1 = target['mood'][target['encode']==int(max_prob_indx[1][0])]
    mood2 = target['mood'][target['encode']==int(sec_prob_indx[1][0])]
    mood3 = target['mood'][target['encode']==int(third_prob_indx[1][0])]
    mood4 = target['mood'][target['encode']==int(fourth_prob_indx[1][0])]

    moods = []
    moods.append(mood1.values)
    moods.append(mood2.values)
    moods.append(mood3.values)
    moods.append(mood4.values)

    energy_calm = False
    happy_sad = False
    final_moods = []
    for mood in moods:
        if((mood[0] == 'Energetic' or mood[0] == 'Calm') and energy_calm == False): 
            final_moods.append(mood[0])
            energy_calm = True
    
        if ( (mood[0] == 'Happy' or mood[0] == 'Sad') and happy_sad == False): 
            final_moods.append(mood[0])
            happy_sad = True

    return final_moods 

if __name__ == "__main__":
    artist = "Brunori Sas"
    song = "Come Stai"
    song_id = search_for_song_id(artist, song)
    moods = predict(song_id)
    print(moods)