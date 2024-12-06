import streamlit as st

# Home page setup
st.set_page_config(page_title="Emotion Detection", layout="wide")
st.title("Emotion Detection")
st.write("Choose how you would like to detect emotion:")

# Buttons for face or voice detection
if st.button("Detect by Face"):
    st.session_state['page'] = 'face'

if st.button("Detect by Voice"):
    st.session_state['page'] = 'voice'

# Navigate to the selected page
if 'page' in st.session_state:
    if st.session_state['page'] == 'face':
        st.experimental_rerun()  # Rerun to load the face detection code
    elif st.session_state['page'] == 'voice':
        st.experimental_rerun()  # Rerun to load the voice detection code
