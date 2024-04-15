import streamlit as st
import joblib
import numpy as np
import torch
import matplotlib.pyplot as plt
from imageio import imread

# Load the model on CPU
#model = torch.load('CNN_LSTM.pt', map_location=torch.device('cpu'))
lasso_model = joblib.load('lasso_best_model.pkl')
def main():
    st.title("Corn Yield Prediction")
    st.write("Select a satellite image to predict the corn yield.")
    # Display the satellite images as interactive elements
    col1, col2, col3, col4= st.columns(4)
    with col1:
        if st.image("assets/17_123.png", caption="Satellite Image 1",):
            if st.button("Select Image 1"):
                # Load the image data
                image_data = np.load("assets/2018_17_123.npy")
                
                # Preprocess the image data (if needed)
                # Perform any necessary preprocessing steps on the image data here
                
                # Convert the image data to a PyTorch tensor and add a batch dimension
                #image_tensor = torch.tensor(image_data, dtype=torch.float32).unsqueeze(0)
                
                # Perform prediction using the loaded model
                # with torch.no_grad():
                #     prediction = model(image_tensor)
                flattened_image = image_data.reshape(1, -1)
                
                # Perform prediction using the Lasso model
                prediction = lasso_model.predict(flattened_image)
                # Display the prediction
                st.write("Yield prediction:", prediction.item())

    with col2:
        if st.image("assets/19_165.png", caption="Satellite Image 2"):
            if st.button("Select Image 2"):
                # Load the image data
                image_data = np.load("assets/2018_19_165.npy")
                
                # Preprocess the image data (if needed)
                # Perform any necessary preprocessing steps on the image data here
                
                # Convert the image data to a PyTorch tensor and add a batch dimension
                #image_tensor = torch.tensor(image_data, dtype=torch.float32).unsqueeze(0)
                
                # Perform prediction using the loaded model
                # with torch.no_grad():
                #     prediction = model(image_tensor)
                flattened_image = image_data.reshape(1, -1)
                
                # Perform prediction using the Lasso model
                prediction = lasso_model.predict(flattened_image)
                # Display the prediction
                st.write("Yield prediction:", prediction.item())


    with col3:
        if st.image("assets/22_107.png", caption="Image 3"):
            if st.button("Select Image 3"):    
                # Load the image data
                image_data = np.load("assets/2018_22_107.npy")
                
                # Preprocess the image data (if needed)
                # Perform any necessary preprocessing steps on the image data here
                
                # Convert the image data to a PyTorch tensor and add a batch dimension
                #image_tensor = torch.tensor(image_data, dtype=torch.float32).unsqueeze(0)
            
                # Perform prediction using the loaded model
                # with torch.no_grad():
                #     prediction = model(image_tensor)
                flattened_image = image_data.reshape(1, -1)
                
                # Perform prediction using the Lasso model
                prediction = lasso_model.predict(flattened_image)
                # Display the prediction
                st.write("Yield prediction:", prediction.item())
    with col4:
        if st.image("assets/31_109.png", caption="Image 3"):
            if st.button("Select Image 4"):    
                # Load the image data
                image_data = np.load("assets/2018_31_109.npy")
                
                # Preprocess the image data (if needed)
                # Perform any necessary preprocessing steps on the image data here
                
                # Convert the image data to a PyTorch tensor and add a batch dimension
                #image_tensor = torch.tensor(image_data, dtype=torch.float32).unsqueeze(0)
            
                # Perform prediction using the loaded model
                # with torch.no_grad():
                #     prediction = model(image_tensor)
                flattened_image = image_data.reshape(1, -1)
                
                # Perform prediction using the Lasso model
                prediction = lasso_model.predict(flattened_image)
                # Display the prediction
                st.write("Yield prediction:", prediction.item())



if __name__=='__main__':
    main()