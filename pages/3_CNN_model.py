import streamlit as st
import pickle
import numpy as np
import torch

# Load the model on CPU
model = torch.load('CNN_LSTM.pt', map_location=torch.device('cpu'))


def main():
    st.title("Corn Price Prediction")
    st.write("Select a satellite image to predict the corn yield.")
    # Display the satellite images as interactive elements
    col1, col2, col3 = st.columns(3)
    with col1:
        if st.image("assets/1.png", caption="Image 1"):
            if st.button("Image 1"):
                # Load the image data
                image_data = np.load("assets/2018_13_147.npy")
                
                # Preprocess the image data (if needed)
                # Perform any necessary preprocessing steps on the image data here
                
                # Convert the image data to a PyTorch tensor and add a batch dimension
                image_tensor = torch.tensor(image_data, dtype=torch.float32).unsqueeze(0)
                
                # Perform prediction using the loaded model
                with torch.no_grad():
                    prediction = model(image_tensor)
                
                # Display the prediction
                st.write("Prediction:", prediction.item())

    with col2:
        if st.image("assets/2.jpeg", caption="Image 2"):
            if st.button("Image 2"):
                # Load the image data
                image_data = np.load("assets/2018_13_131.npy")
                
                # Preprocess the image data (if needed)
                # Perform any necessary preprocessing steps on the image data here
                
                # Convert the image data to a PyTorch tensor and add a batch dimension
                image_tensor = torch.tensor(image_data, dtype=torch.float32).unsqueeze(0)
                
                # Perform prediction using the loaded model
                with torch.no_grad():
                    prediction = model(image_tensor)
                
                # Display the prediction
                st.write("Prediction:", prediction.item())


    with col3:
        if st.image("assets/3.jpeg", caption="Image 3"):
            if st.button("Image 3"):    
                # Load the image data
                image_data = np.load("assets/2018_13_115.npy")
                
                # Preprocess the image data (if needed)
                # Perform any necessary preprocessing steps on the image data here
                
                # Convert the image data to a PyTorch tensor and add a batch dimension
                image_tensor = torch.tensor(image_data, dtype=torch.float32).unsqueeze(0)
            
                # Perform prediction using the loaded model
                with torch.no_grad():
                    prediction = model(image_tensor)
                
                # Display the prediction
                st.write("Prediction:", prediction.item())


if __name__=='__main__':
    main()