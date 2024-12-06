import os
import pickle
import numpy as np
import librosa
from tensorflow.keras.models import model_from_json
import streamlit as st
from st_audiorec import st_audiorec

# Load the CNN model architecture from JSON file
json_file = open('CNN_model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)

# Load the CNN model weights
loaded_model.load_weights("CNN_model_weights.h5")
print("Loaded CNN model from disk")

# Load the scaler and encoder for feature scaling and label encoding
with open('scaler2.pickle', 'rb') as f:
    scaler2 = pickle.load(f)

with open('encoder2.pickle', 'rb') as f:
    encoder2 = pickle.load(f)

print("Loaded scaler and encoder")

# Define functions for feature extraction
def zcr(data, frame_length, hop_length):
    zcr = librosa.feature.zero_crossing_rate(y=data, frame_length=frame_length, hop_length=hop_length)
    return np.squeeze(zcr)

def rmse(data, frame_length=2048, hop_length=512):
    rmse = librosa.feature.rms(y=data, frame_length=frame_length, hop_length=hop_length)
    return np.squeeze(rmse)

def mfcc(data, sr, frame_length=2048, hop_length=512, n_mfcc=13, flatten=True):
    mfcc = librosa.feature.mfcc(y=data, sr=sr, n_fft=frame_length, hop_length=hop_length, n_mfcc=n_mfcc)
    return np.squeeze(mfcc.T) if not flatten else np.ravel(mfcc.T)

def extract_features(data, sr=22050, frame_length=2048, hop_length=512, num_mfcc=13, total_features=1620):
    result = np.array([])

    # Extract ZCR and RMSE
    result = np.hstack((result, zcr(data, frame_length, hop_length), rmse(data, frame_length, hop_length)))

    # Extract MFCCs
    mfcc_features = mfcc(data, sr, frame_length, hop_length, n_mfcc=num_mfcc)
    remaining_features = total_features - len(result)
    
    # Trim or pad MFCC features
    if len(mfcc_features) > remaining_features:
        mfcc_features = mfcc_features[:remaining_features]
    elif len(mfcc_features) < remaining_features:
        pad_length = remaining_features - len(mfcc_features)
        mfcc_features = np.pad(mfcc_features, ((0, pad_length), (0, 0)))

    result = np.hstack((result, mfcc_features))
    return result

def get_predict_feat(path):
    d, s_rate = librosa.load(path, duration=6, offset=0.6)
    res = extract_features(d)
    result = np.array(res).reshape(1, -1)
    i_result = scaler2.transform(result)
    final_result = np.expand_dims(i_result, axis=2)
    return final_result

emotions = {1: 'Angry', 2: 'Calm', 3: 'Happy', 4: 'Sad', 5: 'Neutral', 6: 'Fear', 7: 'Disgust', 8: 'Surprise'}

def prediction(path):
    res = get_predict_feat(path)
    predictions = loaded_model.predict(res)
    y_pred = encoder2.inverse_transform(predictions)
    return y_pred[0][0]

def save_wav_file(audio_data):
    if not os.path.exists('sound'):
        os.makedirs('sound')
    file_path = os.path.join('sound', 'recorded_audio.wav')
    with open(file_path, 'wb') as f:
        f.write(audio_data)
    return file_path

def audiorec_demo_app():
    #st.set_page_config(page_title="Sound Emotion Detection", page_icon="🎤")
    st.markdown(
        """
        <style>
            body {
                background-image: url('https://example.com/your-background-image.jpg');
                background-size: cover;
                color: white;
            }
            .container {
                background-color: rgba(0, 0, 0, 0.7);
                padding: 20px;
                border-radius: 10px;
                text-align: center;
                box-shadow: 0 0 10px rgba(0, 0, 0, 0.5);
            }
            h1 {
                font-size: 2.5em;
                margin-bottom: 10px;
            }
            p {
                font-size: 1.2em;
            }
        </style>
        <div class='container'>
            <h1>Sound Emotion Detection</h1>
            <p>Click the button below to start recording your audio (minimum 5 seconds).</p>
            <img src='https://tse4.mm.bing.net/th?id=OIP.OdXfQIjYDc2vOFimTk4uCAHaEK&pid=Api&P=0&h=180' alt='Microphone' style='width:200px;height:200px;'>
        </div>
        """,
        unsafe_allow_html=True
    )

    wav_audio_data = st_audiorec()

    if wav_audio_data is not None:
        if len(wav_audio_data) / 44100 > 5:
            file_path = save_wav_file(wav_audio_data)
            st.success("Recording stopped. Ready for prediction!")
            
            if st.button("Predict"):
                with st.spinner("Predicting..."):
                    try:
                        predicted_emotion = prediction(file_path)
                        st.write(f'**Predicted Emotion:** {predicted_emotion}')
                    except Exception as e:
                        st.error(f'Error during prediction: {str(e)}')
        else:
            st.error('Please record at least 5 seconds.')

if __name__ == '__main__':
    audiorec_demo_app()
