# Import the libraries
import os
import streamlit as st
from main import AlzheimerPrediction
from tempfile import NamedTemporaryFile
import warnings
warnings.filterwarnings("ignore")  # hide deprication warnings which directly don't affect the working of the application

# set the page layout to wide
# set the page layout
# set some pre-defined configurations for the page, such as the page title, logo-icon, page loading state 
# (whether the page is loaded automatically or you need to perform some action for loading)
st.set_page_config(
    page_title="Alzheimer Disease Detection",
    page_icon = "sample_mri.png",
    initial_sidebar_state = 'auto'
)

# hide the part of the code, as this is just for adding some custom CSS styling but not a part of the main idea 
hide_streamlit_style = """
	<style>
  #MainMenu {visibility: hidden;}
	footer {visibility: hidden;}
  </style>
"""
st.markdown(hide_streamlit_style, unsafe_allow_html=True) # hide the CSS code from the screen as they are embedded in markdown text. Also, allow streamlit to unsafely process as HTML

# add the logo and title of the application

with st.sidebar:
    try:
        st.image('sample_mri.png')
    except:
        st.write("Please upload an image")
    st.title("Alzheimer Disease Detection")
    st.subheader("Accurate detection of neuro-degenarative disease Alzheimer from 3D MRI volumes. This helps an user to easily detect the disease and identify it's cause.")

st.write("""
         # Alzheimer Disease Detection
         """
         )

file = st.file_uploader("", type=[".nii"])
        
if file is None:
    st.text("Please upload an MRI volume file")
else:
    with open(os.getcwd() + '/Uploaded.nii', 'wb') as f: 
        f.write(file.getvalue())
    image_path = os.getcwd() + '/Uploaded.nii'
    AlzheimerPrediction = AlzheimerPrediction()
    scores, label = AlzheimerPrediction.prediction(image_path)
    os.remove(image_path)
    print(scores, label)
    st.sidebar.subheader("Prediction: ")
    st.sidebar.error("Accuracy : " + str(max(scores)*100) + " %")

    class_names = ['Normal', 'Alzheimer']

    
    if class_names[label] == 'Normal':
        string = "No Alzheimer Disease Detected"
        st.balloons()
        st.sidebar.success(string)

    elif class_names[label] == 'Alzheimer':
        string = "Alzheimer Disease Detected"
        st.sidebar.warning(string)
        
