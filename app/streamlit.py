import streamlit as st
import requests

API = 'http://localhost:8000'

st.title('Cognitive Dissonance Detector — Demo')
text = st.text_area('Enter text', 'I love my job, but I dread Mondays and feel exhausted.')
if st.button('Analyze'):
    try:
        r = requests.post(API + '/predict', json={'text': text}).json()
        st.metric('Dissonance score', f"{r['dissonance']:.3f}")
        st.write('Heuristic score:', r['heuristic'])
        st.write('Meta features:')
        st.json(r['meta'])
    except Exception as e:
        st.error('Could not contact API. Make sure uvicorn app.api:app is running')
