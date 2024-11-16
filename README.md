# Alzheimer Disease Detection with GCN

This project is a deep learning application for detecting Alzheimer’s disease from 3D MRI volumes. It leverages Graph Convolutional Networks (GCNs) to classify images into either Normal or Alzheimer categories. The application is designed to be user-friendly, using a Streamlit-based web interface for inference. Below are the details for setup, usage, and implementation.

---

## Features
- 3D Convolutional Network: Extracts features from 3D MRI volumes.
- Graph Convolutional Network (GCN): Processes spatial relationships in patches of the 3D image to classify Alzheimer’s.
- User-friendly Interface: Upload .nii files through a Streamlit app for predictions.
- Interactive Visuals: Provides real-time feedback on classification results.

---

## Installation
### Prerequisites
Ensure the following are installed:

- Python >= 3.8
- pip
- PyTorch (with CUDA if using GPU)
- Required Python libraries: torch_geometric, nibabel, scikit-learn, scipy, streamlit

### Steps
1. Clone the repository:
```
git clone https://github.com/your-repo/alzheimer-detection.git
cd alzheimer-detection
```
2. Install dependencies:
```
pip install -r requirements.txt
```
3. Ensure PyTorch and PyTorch Geometric are installed for your system. For example:
```
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
```
4. Place the trained model (final_model_2023-09-16 14:41:14.661342.pt) in the same directory as `main.py`.

---
## Usage
### Running the Application
Video Demo - https://youtu.be/4lKZybx8s7Q 

1. Start the Streamlit app:
```
streamlit run App.py
```

2. Open the browser and go to the displayed URL (e.g., http://localhost:8501).

3. Upload a .nii MRI volume file using the interface.

4. View the prediction results on the sidebar:
- Normal: No Alzheimer detected.
- Alzheimer: Indicates the presence of Alzheimer’s disease.


---
## File Structure
```
.
├── main.py              # Core model implementation
├── App.py               # Streamlit-based web application
├── requirements.txt     # Dependencies
├── README.md            # Documentation
├── sample_mri.png       # Logo/icon for the app
├── Uploaded.nii         # Placeholder for uploaded MRI files
└── final_model_2023-09-16 14:41:14.661342.pt # Pre-trained model
```

---
## Key Components
### Model (GCN)
Defined in main.py, the GCN class integrates:
- 3D Convolutional Layers: Extract spatial features from MRI images.
- Graph Convolutional Layers: Utilize adjacency matrices to process graph structures representing image patches.
- Fully Connected Layers: Classify extracted features.

### Graph Construction
- Image patches are extracted and connected based on spatial proximity.
- The adjacency matrix is built using the K-Nearest Neighbors (KNN) graph.

### Streamlit Application
- Users can upload .nii files and get predictions.
- Real-time display of accuracy and diagnosis.


---
## Notes
- The app is currently designed to run on the CPU. Update the device variable in main.py to 'cuda' for GPU support if available.
- Ensure `.nii` MRI files are correctly formatted and preprocessed.

---
## Future Enhancements
- Expand Dataset: Improve model accuracy with more diverse data.
- Visualization: Add MRI visualization within the app.
- Explainability: Introduce feature importance or saliency maps for predictions.

---
## Credits
Model Design: 
Research: Developed by combining deep learning and neuroimaging research techniques.
Frontend: Powered by Streamlit for a smooth user experience.

----
## License
This project is open-source and available under the MIT License.
