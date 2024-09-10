import os
import numpy as np
import cv2
from flask import Flask, request, render_template, redirect, url_for
from werkzeug.utils import secure_filename
from keras_facenet import FaceNet

app = Flask(__name__)
UPLOAD_FOLDER = 'static/uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

model = FaceNet().model
database = np.load('models/lfw_embeddings.npy')
labels = np.load('models/lfw_labels.npy')

# Create a dictionary from labels and embeddings
database = {label: embedding for label, embedding in zip(labels, database)}

# Normalize the stored embeddings ONCE when loading the database
for label in database:
    norm = np.linalg.norm(database[label], ord=2)
    database[label] = database[label] / norm

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def load_image(image_path):
    image = cv2.imread(image_path)
    image = cv2.resize(image, (160, 160))
    image = image.astype(np.float32) / 255.0
    return image

def image_to_embedding(image, model):
    embedding = model.predict(image[np.newaxis, ...])
    # Normalize the embedding ONCE after generation
    embedding = embedding.flatten()  # Flatten the embedding
    embedding /= np.linalg.norm(embedding, ord=2)
    return embedding

def compare_embeddings(embedding_1, embedding_2, threshold=0.6):
    dist = np.linalg.norm(embedding_1 - embedding_2)
    return dist < threshold

def recognize_face(image, model, database, threshold=1):
    image_emb = image_to_embedding(image, model)
    
    # Ensure the new embedding is normalized and flattened
    print(f"Image embedding shape: {image_emb.shape}")
    print(f"Image embedding values: {image_emb}")

    min_dist = float('inf')
    name = "No Match Found for you"
    
    for label, embed in database.items():
        # Distance calculation
        dist = np.linalg.norm(embed - image_emb)
        print(f"Comparing to {label}: Distance = {dist}")
        
        if dist < min_dist and dist < threshold:
            min_dist = dist
            name = label
            
    return name

@app.route('/', methods=['GET', 'POST'])
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return redirect(request.url)
    
    file = request.files['file']
    
    if file.filename == '':
        return redirect(request.url)
    
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        image = load_image(filepath)
        recognized_name = recognize_face(image, model, database)
        
        return render_template('index.html', result=recognized_name, image_url=url_for('static', filename='uploads/' + filename))
    
    return redirect(request.url)

if __name__ == '__main__':
    if not os.path.exists(UPLOAD_FOLDER):
        os.makedirs(UPLOAD_FOLDER)
    app.run(debug=True)
