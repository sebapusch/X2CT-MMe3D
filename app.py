import requests
import streamlit as st
import numpy as np
import matplotlib.pyplot as plt

INFERENCE_ENDPOINT = 'http://localhost:8000/predict'


if 'diagnosis' not in st.session_state.keys():
    st.session_state.diagnosis = None
    st.session_state.path_raw = None
    st.session_state.path_grad = None
    st.session_state.slice_idx = [0, 0, 0, 0]

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

            st.session_state.diagnosis = response['diagnosis']
            st.session_state.path_raw  = response['path_raw_volume']
            st.session_state.path_grad = response['path_grad_volume']

        except requests.exceptions.HTTPError as e:
            diagnosis = None
            st.error(e)
        except Exception as e:
            diagnosis = None
            st.error(e)


if st.session_state.diagnosis is not None:
    with st.container(border=1):
        st.metric(label="Diagnosis", value=st.session_state.diagnosis.title())

    cols = st.columns(2)
    volumes = [st.session_state.path_raw, st.session_state.path_grad]

    d, h, w = (128, 128, 128)

    st.session_state.slice_idx[0] = st.slider('Axial', 0, d - 1, d // 2,)
    for i in range(2):
        volume = np.load(volumes[i])
        with cols[i]:
            fig, ax = plt.subplots()
            ax.imshow(volume[st.session_state.slice_idx[0], :, :], cmap='gray')
            ax.axis('off')
            st.pyplot(fig)

    cols = st.columns(2)
    st.session_state.slice_idx[1] = st.slider('Sagittal', 0, w - 1, w // 2,)
    for i in range(2):
        volume = np.load(volumes[i])
        with cols[i]:
            fig, ax = plt.subplots()
            ax.imshow(volume[:, st.session_state.slice_idx[1], :], cmap='gray')
            ax.axis('off')
            st.pyplot(fig)

    cols = st.columns(2)
    st.session_state.slice_idx[2] = st.slider('Coronal', 0, h - 1, h // 2)
    for i in range(2):
        volume = np.load(volumes[i])
        with cols[i]:
            fig, ax = plt.subplots()
            ax.imshow(volume[:, :, st.session_state.slice_idx[2]], cmap='gray')
            ax.axis('off')
            st.pyplot(fig)

