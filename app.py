import requests
import streamlit as st
import numpy as np
import matplotlib.pyplot as plt

INFERENCE_ENDPOINT = 'http://localhost:8000/predict'

st.session_state.diagnosis = 'disease'

frontal = st.file_uploader('Frontal')
lateral = st.file_uploader('Lateral')
submit  = st.button('Submit', disabled=frontal is None or lateral is None)

if submit:
    with st.spinner('Running inference...'):
        files = {
            'frontal': (frontal.name, frontal, frontal.type),
            'lateral': (lateral.name, lateral, lateral.type),
        }

        try:
            response = requests.post(INFERENCE_ENDPOINT, files=files).json()
            print(response)
            st.session_state.diagnosis = response['diagnosis']

        except requests.exceptions.HTTPError as e:
            diagnosis = None
            st.error(e)
        except Exception as e:
            diagnosis = None
            st.error(e)

if st.session_state.diagnosis is not None:
    with st.container(border=1):
        st.text(f'Diagnosis: {st.session_state.diagnosis}')

    volume = np.load('./data/app/volume.npy')
    slice_idx = st.slider("Slice Index", 0, volume.shape[1] - 1, volume.shape[1] // 2)
    fig, ax = plt.subplots()
    ax.imshow(volume[0, slice_idx, :, :], cmap='gray')
    ax.axis('off')
    st.pyplot(fig)

