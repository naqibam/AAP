from flask import Flask, jsonify, request,render_template,redirect, url_for,session, flash
import numpy as np
import tensorflow as tf
import cv2
from PIL import Image
import keras
import os
from tensorflow.keras.models import load_model
from flask_sqlalchemy import SQLAlchemy
from datetime import datetime 
import base64
from werkzeug.security import generate_password_hash, check_password_hash

from fileinput import filename
import pandas as pd
import nltk
import numpy as np
import scipy
from werkzeug.utils import secure_filename
import random
import math
import re
from textblob import TextBlob
import string
import tensorflow as tf
import cv2
import numpy as np
from transformers import TFAutoModelForSequenceClassification
from transformers import AutoTokenizer

# nltk
nltk.download('stopwords')
nltk.download('punkt')
from nltk.corpus import stopwords
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.stem import PorterStemmer
nltk.download("wordnet")
nltk.download("omw-1.4")

UPLOAD_FOLDER = os.path.join('staticFiles', 'uploads')
ALLOWED_EXTENSIONS = {'csv'}

def clean(data):
    data = data.translate(str.maketrans('', '', string.punctuation))
    print(data)
    data = data.lower()
    print(data)
    stop = stopwords.words('english')
    data = ''.join([x for x in re.split(r'(\W+)', data) if x not in stop])
    print(data)
    data = str(TextBlob(data).correct())
    print(data)
    st = WordNetLemmatizer()
    data = st.lemmatize(data)
    print(data)
    return data

app = Flask(__name__)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.secret_key = 'supersecretkey'

NaqSentMdl = TFAutoModelForSequenceClassification.from_pretrained("finetuned_model")
tokenizer = AutoTokenizer.from_pretrained('distilbert-base-uncased')
# Define constants
IMAGE_SIZE = 150

# Load the pre-trained model
model = load_model('mymodel1.h5')

second_model_path = './complete_model.keras'
if not os.path.exists(second_model_path):
    raise FileNotFoundError(f"Model file not found at {second_model_path}")

second_model = tf.keras.models.load_model(second_model_path)

# Define constants for the second model
UNIFORM_SIZE = (128, 128)

# Mapping of predicted classes to messages for the second model
gesture_messages = {
    0: "few minutes",
    1: "more than 10 minutes",
    2: "skip the class"
}


app.config['SQLALCHEMY_DATABASE_URI'] = 'mysql://aap:mysql@localhost:3306/aap'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

db = SQLAlchemy(app)

class Gesture(db.Model):
    __tablename__ = 'gesture'
    id = db.Column(db.Integer, primary_key=True, autoincrement=True)
    gesture = db.Column(db.Integer, nullable=False)
    message = db.Column(db.String(255), nullable=False)
    

    def __repr__(self):
        return f'<Gesture {self.gesture}: {self.message}>'

# Define the Message model
class Message(db.Model):
    __tablename__ = 'message'
    id = db.Column(db.Integer, primary_key=True, autoincrement=True)
    message = db.Column(db.String(255), nullable=False)
    duration = db.Column(db.String(50),nullable=True)
    name = db.Column(db.String(50), nullable=False)
    senttime = db.Column(db.DateTime, nullable=False, default=datetime.utcnow)

    def __repr__(self):
        return f'<Message {self.id}: {self.name} sent {self.message} at {self.senttime}>'

class Staff(db.Model):
    __tablename__ = 'staff'
    id = db.Column(db.Integer, primary_key=True, autoincrement=True)
    username = db.Column(db.String(50), nullable=False, unique=True)
    password_hash = db.Column(db.String(255), nullable=False)

    def set_password(self, password):
        self.password_hash = generate_password_hash(password)

    def check_password(self, password):
        return check_password_hash(self.password_hash, password)

# Define image processing function
def process_image(img):
    image = []
    # Convert to grayscale
    pil_image = Image.fromarray(img)
    pil_image = pil_image.convert('L')
    # Resize image
    resized_image = pil_image.resize((IMAGE_SIZE, IMAGE_SIZE), Image.BICUBIC)
    resized_image = np.array(resized_image)
    # Normalize image
    resized_image = resized_image.astype('float32') / 255.0
    image.append(resized_image)
    image = np.array(image)
    readyimage = image[0]
    readyimage = readyimage.reshape((1,) + readyimage.shape)
    return readyimage


# Define image processing function for the second model
def process_image_for_second_model(img):
    image = Image.fromarray(img)
    image = image.convert('RGB')
    image = image.resize(UNIFORM_SIZE, Image.LANCZOS)
    image = np.array(image) / 255.0
    image = np.expand_dims(image, axis=0)
    return image

# Define route for image classification
@app.route('/classifyImage', methods=['POST'])
def predict_image():
    if 'capturedImage' in request.form:
        # Decode the base64 image data
        image_data = request.form['capturedImage']
        image_data = image_data.split(',')[1]
        imgbytes = base64.b64decode(image_data)
        img = np.frombuffer(imgbytes, np.uint8)
        img = cv2.imdecode(img, cv2.IMREAD_COLOR)
    else:
        # Load image from file
        filestream = request.files['file'].read()
        imgbytes = np.frombuffer(filestream, np.uint8)
        img = cv2.imdecode(imgbytes, cv2.IMREAD_COLOR)
    
    # Process the image
    processed_image = process_image(img)
    # Process the image for the second model
    processed_image_second_model = process_image_for_second_model(img)
    
    # Predict and return result
    prediction = model.predict(processed_image)
    predicted_class = np.argmax(prediction)

    match predicted_class:
        case 0:
            predicted_class = 1
        case 1:
            predicted_class = 2
        case 2:
            predicted_class = 3
        case 3:
            predicted_class = "Others"

    # Predict and return result for the second model
    prediction_second_model = second_model.predict(processed_image_second_model)
    predicted_class_second_model = np.argmax(prediction_second_model, axis=1)[0]
    message_second_model = gesture_messages.get(predicted_class_second_model, "Unknown hand raised")

    if predicted_class != "Others" and message_second_model !="Unknown hand raised":
        # Find the message in the Gesture table
        gesture = Gesture.query.filter_by(id=predicted_class).all()
        if gesture:
            name = request.form.get('userName')
            message_text = gesture[0].message
            # Create a new row in the Message table
            new_message = Message(message=message_text, name=name,duration = message_second_model)
            db.session.add(new_message)
            db.session.commit()
    else:
        message_text = "Invalid hand gesture"
    


    

    return render_template('handgesture.html', prediction=message_text,prediction_second_model=message_second_model)

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        staff = Staff.query.filter_by(username=username).first()
        if staff and staff.check_password(password):
            session['authenticated'] = True
            return redirect(url_for('home'))
        else:
            flash('Invalid username or password')
            return redirect(url_for('login'))
    return render_template('login.html')

@app.route('/logout')
def logout():
    session.pop('authenticated', None)
    return redirect(url_for('home'))

@app.route('/handgesture')
def handgesture():
    return render_template('handgesture.html')

@app.route('/customize', methods=['GET'])
def customize():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        staff = Staff.query.filter_by(username=username).first()
        if staff and staff.check_password(password):
            session['authenticated'] = True
            return redirect(url_for('customize'))
        else:
            flash('Invalid username or password')
            return redirect(url_for('home'))
    
    if 'authenticated' in session and session['authenticated']:
        gestures = Gesture.query.all()
        return render_template('customize.html', gestures=gestures)
    else:
        return redirect(url_for('home'))
    
@app.route('/message', methods=['GET'])
def MessageTable():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        staff = Staff.query.filter_by(username=username).first()
        if staff and staff.check_password(password):
            session['authenticated'] = True
            return redirect(url_for('MessageTable'))
        else:
            flash('Invalid username or password')
            return redirect(url_for('home'))

    if 'authenticated' in session and session['authenticated']:
        messages = Message.query.all()
        return render_template('message.html', messages=messages)
    else:
        return render_template('login.html')

@app.route('/deletemessage/<int:id>', methods=['POST'])
def DeleteMessage(id):
    # Find the message by ID
    message = Message.query.get(id)
    if message:
        # Delete the message
        db.session.delete(message)
        db.session.commit()
        return redirect(url_for('MessageTable'))
    else:
        return "Message not found", 404


def insert_default_gestures():
    row_count = Gesture.query.count()
    if row_count == 0:
        default_gestures = [
            Gesture(gesture=1, message="Gesture 1 message"),
            Gesture(gesture=2, message="Gesture 2 message"),
            Gesture(gesture=3, message="Gesture 3 message")
        ]
        db.session.bulk_save_objects(default_gestures)
        db.session.commit()

def insert_default_staff():
    row_count = Staff.query.count()
    if row_count == 0:
        default_staff = Staff(username='admin')
        default_staff.set_password('P@ssw0rd')
        db.session.add(default_staff)
        db.session.commit()

@app.route('/updateGestures', methods=['POST'])
def updateGestures():
    gestures = Gesture.query.all()
    for gesture in gestures:
        new_message = request.form.get(f'gesture_{gesture.id}_message')
        if new_message:
            gesture.message = new_message
    db.session.commit()

    return redirect(url_for('customize'))

#############################################
############## NAQIB ROUTES #################
#############################################

@app.route('/uploadCSV', methods=['GET', 'POST'])
def uploadFile():
    if request.method == 'POST':
      # upload file flask
        f = request.files.get('file')

        # Extracting uploaded file name
        data_filename = secure_filename(f.filename)

        f.save(os.path.join(app.config['UPLOAD_FOLDER'], data_filename))

        session['uploaded_data_file_path'] = os.path.join(app.config['UPLOAD_FOLDER'], data_filename)

        return render_template('SentimentForm2.html')
    return render_template("SentimentForm.html")

@app.route('/show_data')
def showData():
    # Uploaded File Path
    data_file_path = session.get('uploaded_data_file_path', None)
    # read csv
    uploaded_df = pd.read_csv(data_file_path, encoding='unicode_escape')
    # Converting to html Table
    uploaded_df_html = uploaded_df.to_html()
    return render_template('show_csv_data.html', data_var=uploaded_df_html)

@app.route('/SentimentForm')
def SentimentForm():
    return render_template('SentimentForm.html')

@app.route("/sentiment")
def sentiment():
    Positive = 0
    Negative = 0
    # Uploaded File Path
    data_file_path = session.get('uploaded_data_file_path', None)
    # read csv
    uploaded_df = pd.read_csv(data_file_path, encoding='unicode_escape')
    predictions = pd.DataFrame(columns=['Prediction'])
    #target = ["Negative", "Positive"]
    for index, row in uploaded_df.iterrows():
        feedback_cleaned = clean(row['comment'])
        inputs = tokenizer(feedback_cleaned, return_tensors="tf")
        output = NaqSentMdl(inputs)
        pred_prob = tf.nn.softmax(output.logits, axis=-1)
        pred = np.argmax(pred_prob)
        if pred == 1:
            new_row = {"Prediction": "Positive" }
            predictions = pd.concat([predictions, pd.DataFrame([new_row])], ignore_index=True)
            Positive += 1
        else:
            new_row = {"Prediction": "Negative" }
            predictions = pd.concat([predictions, pd.DataFrame([new_row])], ignore_index=True)
            Negative += 1
    uploaded_df["Prediction"] = predictions["Prediction"]
    # Converting to html Table
    uploaded_df_html = uploaded_df.to_html()
    return render_template('show_csv_data.html',
                           data_var=uploaded_df_html,
                           act1="Positive",
                           act2="Negative",
                           t1 = Positive,
                           t2 = Negative,
                           ht=500, wt=800,
                           title="Feedback")

########################################################
########################################################
########################################################

if __name__ == '__main__':
    with app.app_context():
        db.create_all()
        insert_default_gestures()
        insert_default_staff()
 
    app.run(debug=True)
