import pickle
import streamlit as st
import pandas as pd
import librosa
import numpy as np
import os
import numpy as np
from pydub import AudioSegment
from sklearn.preprocessing import StandardScaler
#import librosa
#from spleeter.separator import Separator

@st.cache()

# loading the trained model
def load_model(clf_name):
    pickle_in = open(clf_name, 'rb') 
    classifier = pickle.load(pickle_in)
    return classifier
  
# defining the function which will make the prediction using the data which the user inputs 
def prediction(data,clf_name):   
    # Making predictions
    scaler = StandardScaler()
    data_transformed = scaler.fit_transform(data)
    df = pd.DataFrame(data_transformed)
    
    classifier = load_model(clf_name)
    prediction = list(classifier.predict(df))
    predict_proba = classifier.predict_proba(df)
    result = max(set(prediction), key = prediction.count)
    return result, np.mean(np.amax(predict_proba, axis=1))

# function to calculate the mean and variance
def mean_var_calculator(x_list):
    n_array = np.array(x_list)
    return list(n_array.mean(axis=1)), list(n_array.var(axis=1))

# function to extract the features from wav file
def features(path):

    x , sr = librosa.load(path)
    tempo = librosa.beat.tempo(x, sr=sr)
    zero_crossings = sum(librosa.zero_crossings(x, pad=False))
    
    n_sb = librosa.util.normalize(librosa.feature.spectral_bandwidth(x, sr=sr)[0])
    spectral_bandwidth_mean=n_sb.mean()
    spectral_bandwidth_var=n_sb.var()
        
    n_sc = librosa.util.normalize(librosa.feature.spectral_contrast(x, sr=sr)[0])
    spectral_contrast_mean=n_sc.mean()
    spectral_contrast_var=n_sc.var()
    
    n_scc = librosa.util.normalize(librosa.feature.spectral_centroid(x, sr=sr)[0])
    spectral_centroids_mean=n_scc.mean()
    spectral_centroids_var=n_scc.var()
        
    n_sr = librosa.util.normalize(librosa.feature.spectral_rolloff(x, sr=sr)[0])
    spectral_rolloff_mean=n_sr.mean()
    spectral_rolloff_var=n_sr.var()
    
    n_sf = librosa.util.normalize(librosa.feature.spectral_flatness(x)[0])
    spectral_flatness_mean=n_sf.mean()
    spectral_flatness_var=n_sf.var()
    
    chroma_stft_mean, chroma_stft_var  = mean_var_calculator(librosa.feature.chroma_stft(x, sr=sr))
    chroma_cqt_mean, chroma_cqt_var = mean_var_calculator(librosa.feature.chroma_cqt(x, sr=sr))
    chroma_cens_mean, chroma_cens_var = mean_var_calculator(librosa.feature.chroma_cens(x, sr=sr))
    mfcc_mean, mfcc_var = mean_var_calculator(librosa.feature.mfcc(x, sr = sr))
    
    features = [tempo[0], zero_crossings, spectral_bandwidth_mean, spectral_bandwidth_var, spectral_contrast_mean, spectral_contrast_var,
            spectral_centroids_mean, spectral_centroids_var, spectral_rolloff_mean, spectral_rolloff_var, spectral_flatness_mean, spectral_flatness_var ]
    features.extend(chroma_stft_mean)
    features.extend(chroma_stft_var)
    features.extend(chroma_cqt_mean)
    features.extend(chroma_cqt_var)
    features.extend(chroma_cens_mean)
    features.extend(chroma_cens_var)
    features.extend(mfcc_mean)
    features.extend(mfcc_var)
    
    return features

# function to extract features from 5 seconds window of a complete song
def extract_features(path):
    columns=['tempo','zero_crossings','spectral_bandwidth_mean','spectral_bandwidth_var','spectral_contrast_mean','spectral_contrast_var',
         'spectral_centroids_mean','spectral_centroids_var','spectral_rolloff_mean','spectral_rolloff_var','spectral_flatness_mean','spectral_flatness_var',
         
         'chroma_stft1_mean', 'chroma_stft2_mean', 'chroma_stft3_mean', 'chroma_stft4_mean', 'chroma_stft5_mean', 'chroma_stft6_mean', 'chroma_stft7_mean', 
         'chroma_stft8_mean', 'chroma_stft9_mean', 'chroma_stft10_mean', 'chroma_stft11_mean', 'chroma_stft12_mean', 'chroma_stft1_var', 'chroma_stft2_var', 
         'chroma_stft3_var', 'chroma_stft4_var', 'chroma_stft5_var', 'chroma_stft6_var', 'chroma_stft7_var', 'chroma_stft8_var', 'chroma_stft9_var', 
         'chroma_stft10_var', 'chroma_stft11_var', 'chroma_stft12_var',
         
         'chroma_cqt1_mean', 'chroma_cqt2_mean', 'chroma_cqt3_mean', 'chroma_cqt4_mean', 'chroma_cqt5_mean', 'chroma_cqt6_mean', 'chroma_cqt7_mean', 
         'chroma_cqt8_mean', 'chroma_cqt9_mean', 'chroma_cqt10_mean', 'chroma_cqt11_mean', 'chroma_cqt12_mean', 'chroma_cqt1_var', 'chroma_cqt2_var', 
         'chroma_cqt3_var', 'chroma_cqt4_var', 'chroma_cqt5_var', 'chroma_cqt6_var', 'chroma_cqt7_var', 'chroma_cqt8_var', 'chroma_cqt9_var', 
         'chroma_cqt10_var', 'chroma_cqt11_var', 'chroma_cqt12_var',
         
         'chroma_cens1_mean', 'chroma_cens2_mean', 'chroma_cens3_mean', 'chroma_cens4_mean', 'chroma_cens5_mean', 'chroma_cens6_mean', 'chroma_cens7_mean', 
         'chroma_cens8_mean', 'chroma_cens9_mean', 'chroma_cens10_mean', 'chroma_cens11_mean', 'chroma_cens12_mean','chroma_cens1_var', 'chroma_cens2_var', 
         'chroma_cens3_var', 'chroma_cens4_var', 'chroma_cens5_var', 'chroma_cens6_var', 'chroma_cens7_var', 'chroma_cens8_var', 'chroma_cens9_var', 
         'chroma_cens10_var', 'chroma_cens11_var', 'chroma_cens12_var',
         
         'mfcc1_mean', 'mfcc2_mean', 'mfcc3_mean', 'mfcc4_mean', 'mfcc5_mean', 'mfcc6_mean', 'mfcc7_mean', 'mfcc8_mean', 'mfcc9_mean', 'mfcc10_mean', 
         'mfcc11_mean', 'mfcc12_mean', 'mfcc13_mean', 'mfcc14_mean', 'mfcc15_mean', 'mfcc16_mean', 'mfcc17_mean', 'mfcc18_mean', 'mfcc19_mean', 
         'mfcc20_mean', 'mfcc1_var', 'mfcc2_var', 'mfcc3_var', 'mfcc4_var', 'mfcc5_var', 'mfcc6_var', 'mfcc7_var', 'mfcc8_var', 'mfcc9_var',
         'mfcc10_var', 'mfcc11_var', 'mfcc12_var', 'mfcc13_var', 'mfcc14_var', 'mfcc15_var', 'mfcc16_var', 'mfcc17_var', 'mfcc18_var', 
         'mfcc19_var', 'mfcc20_var']
    songs_features = []
    audio = AudioSegment.from_mp3(path)
    my_bar = st.progress(0)
    for i in range(round(len(audio)/5000)):
        my_bar.progress((i+1)*(1/round(len(audio)/5000)))
        try:
            song = audio[i*5000:(i+1)*5000] 
            out_path = 'output.wav'
            song.export(out_path, format="wav")
            temp = features(out_path)
            songs_features.append(temp)
            os.remove(out_path)
        except:
            song = audio[i*5000:(i+1)*5000] 
            out_path = 'output2.wav'
            song.export(out_path, format="wav")
            temp = features(out_path)
            songs_features.append(temp)
            os.remove(out_path)
    df = pd.DataFrame(songs_features, columns = columns)
    return df

# this is the main function in which we define our webpage  
def main():       
    # front end elements of the web page 
    html_temp = """ 
    <div> 
    <h1 style ="color:black;text-align:center;">Music Genre and Artist Prediction</h1> 
    </div> 
    """
    page_bg_img = '''
    <style>
    body {
        background-image: url("https://upload.wikimedia.org/wikipedia/en/thumb/b/bb/Graduated_Blue_Background.png/700px-Graduated_Blue_Background.png");
        background-size: cover;
    }
    </style>
    '''
    st.markdown(page_bg_img, unsafe_allow_html=True)
    
    result =""
    language =""
    clicked =""
    uploaded_file = None
    
    # display the front end aspect
    st.markdown(html_temp, unsafe_allow_html = True)
    type_prediction = st.selectbox('Select the Type of Prediction',('--Select--','Artist Prediction', 'Genre Prediction'))
    if type_prediction in ['Artist Prediction','Genre Prediction']:
        language = st.selectbox('Select the Language',('--Select--','English', 'Hindi', 'Telugu','Unknown'))
    if language in ['English','Hindi','Telugu','Unknown']:
        uploaded_file = st.file_uploader('File uploader')
    if uploaded_file is not None:
            clicked = st.button("Predict")

    # when 'Predict' is clicked, make the prediction and store it 
    if clicked:
        if uploaded_file is not None:
            file_details = {"FileName":uploaded_file.name,"FileType":uploaded_file.type,"FileSize in bytes":uploaded_file.size}
            st.write('Uploaded File Details:\n', file_details)
            st.audio(uploaded_file.name)
            #separator = Separator('spleeter:2stems')  # Using 2stems configuration.
            #separator.separate_to_file(uploaded_file.name, '', synchronous=False) # Wait for batch to finish.
            #separator.join()
            #vocal_path = uploaded_file.name[:-4] + '/vocals.wav'
            #accompaniment_path =  uploaded_file.name[:-4] + '/accompaniment.wav'
            #st.write(vocal_path, accompaniment_path)
            #st.audio(vocal_path)
            #st.audio(accompaniment_path)
            
        df = extract_features(uploaded_file.name)
        st.write(df)
        if language != 'Unknown':
            clf_name = type_prediction.split()[0] + '_' + language + '.pkl'
        else:
            clf_name = 'All_Genres.pkl'
            
        result,predict_proba = prediction(df,clf_name)
        if predict_proba < 0.50 and type_prediction.split()[0] == 'Genre':
            st.success('Sorry, This is unknown {}. Our application is limited to few {}s only'.format(type_prediction.split()[0],type_prediction.split()[0]))
        else:
            st.success('{} Name: {}'.format(type_prediction.split()[0],result.split('_')[1]))
            st.success('Language: {}'.format(result.split('_')[0]))
            #st.success('Prediction Probability: {:.2f}'.format(predict_proba))
        
                 
if __name__=='__main__': 
    main()