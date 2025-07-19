🖼️ Image Similarity Matcher
A simple web app built with Streamlit and PyTorch (ResNet50) that allows you to:

Upload an image

Compare it with 50+ pre-stored images

Get top 5 visually similar matches with similarity scores

🚀 Features
Upload any .jpg, .jpeg, or .png image

Preprocesses and compares using deep feature vectors

Uses cosine similarity for measuring closeness

Visual interface built entirely with Streamlit

Displays top matches side-by-side with their match score

🧠 How It Works
Loads a pretrained ResNet50 model (without the final classification layer)

Extracts feature vectors from all stored images in the images/ folder

Extracts features from the uploaded image

Compares uploaded image with stored ones using cosine similarity

Displays the top 5 matches with their similarity scores

📁 Project Structure
bash
Copy
Edit
image_search/
├── app.py               # Main Streamlit app
├── images/              # Folder containing 50+ stored reference images
│   ├── img1.jpg
│   ├── img2.jpg
│   └── ...
├── requirements.txt     # Python dependencies (optional but recommended)
└── README.md            # You're reading it!
📦 Installation
Clone this repository:

bash
Copy
Edit
git clone https://github.com/your-username/image-search-streamlit.git
cd image-search-streamlit
(Optional but recommended) Create a virtual environment:

bash
Copy
Edit
python -m venv venv
venv\Scripts\activate  # On Windows
source venv/bin/activate  # On macOS/Linux
Install dependencies:

bash
Copy
Edit
pip install -r requirements.txt
If you don’t have a requirements.txt, just run:

bash
Copy
Edit
pip install streamlit torch torchvision scikit-learn pillow
▶️ Run the App
bash
Copy
Edit
streamlit run app.py
Upload your image from the UI.

See the top 5 most similar images from the images/ folder with scores.

🧪 Sample Output
Uploaded Image	Top 5 Matches

📌 Dependencies
streamlit

torch

torchvision

scikit-learn

Pillow

numpy

🔮 Future Improvements
Use CLIP for better semantic similarity

Integrate FAISS for faster large-scale matching

Add filtering and threshold options

Enable image tagging or categorization

📧 Contact
Created by Aamir Lone
For queries or collaboration: aamirlone004@gmail.com