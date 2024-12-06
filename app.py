import streamlit as st
import os

# This must be the first command in your app
st.set_page_config(page_title="Emotion Detection App", page_icon="😊")

# Define a function for the voice detection app
def run_voice_detection():
    import voice_emotion  # Import the voice emotion detection module
    voice_emotion.audiorec_demo_app()  # Call the function that starts the voice app

# Define a function for the facial detection app
def run_face_detection():
    import facial_emotions  # Import the facial emotion detection module
    facial_emotions.start_capturing(0)  # Adjust based on your face detection function

# Main function for the Streamlit app
def main():
    st.title("Emotion Detection App")

    # Dropdown to choose the detection method
    detection_method = st.selectbox("Select Detection Method", ["Detect by Voice", "Detect by Face"])

    if st.button("Start Detection"):
        if detection_method == "Detect by Voice":
            run_voice_detection()
        elif detection_method == "Detect by Face":
            run_face_detection()

if __name__ == "__main__":
    main()
