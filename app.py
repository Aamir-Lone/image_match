import streamlit as st
from PIL import Image
import os
import torch
from torchvision import models, transforms
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

# ----- Setup -----

st.set_page_config(layout="wide")
st.title("Image Similarity Matcher")

# Load model
model = models.resnet50(pretrained=True)
model = torch.nn.Sequential(*list(model.children())[:-1])
model.eval()

# Image transform
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

# Extract feature vector
def extract_features(image: Image.Image):
    image = transform(image).unsqueeze(0)
    with torch.no_grad():
        features = model(image).squeeze().numpy()
    return features

# Load all stored image features
@st.cache_data
def load_stored_images_and_features(folder="images"):
    features = {}
    for file in os.listdir(folder):
        path = os.path.join(folder, file)
        img = Image.open(path).convert("RGB")
        feat = extract_features(img)
        features[file] = (feat, img)
    return features

stored = load_stored_images_and_features()

# ----- Upload Section -----
uploaded_file = st.file_uploader("Upload an image", type=["jpg", "png", "jpeg"])

if uploaded_file:
    uploaded_img = Image.open(uploaded_file).convert("RGB")
    st.image(uploaded_img, caption="Uploaded Image", width=300)
    
    uploaded_feat = extract_features(uploaded_img)
    
    # ----- Matching -----
    scores = []
    for name, (feat, img) in stored.items():
        score = cosine_similarity([uploaded_feat], [feat])[0][0]
        scores.append((name, score, img))
    
    top_matches = sorted(scores, key=lambda x: x[1], reverse=True)[:5]
    
    st.subheader("Top Matches:")
    cols = st.columns(5)
    for idx, (name, score, img) in enumerate(top_matches):
        with cols[idx]:
            st.image(img, caption=f"{name}\nScore: {score:.2f}", width=150)
