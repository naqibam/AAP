from flask import Flask, jsonify, request,render_template,redirect, url_for
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

app = Flask(__name__)

# Define constants
IMAGE_SIZE = 150

# Load the pre-trained model
model = load_model('mymodel1.h5')

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
    name = db.Column(db.String(50), nullable=False)
    senttime = db.Column(db.DateTime, nullable=False, default=datetime.utcnow)

    def __repr__(self):
        return f'<Message {self.id}: {self.name} sent {self.message} at {self.senttime}>'

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

    if predicted_class != "Others":
        # Find the message in the Gesture table
        gesture = Gesture.query.filter_by(id=predicted_class).all()
        if gesture:
            name = request.form.get('userName')
            message_text = gesture[0].message
            # Create a new row in the Message table
            new_message = Message(message=message_text, name=name)
            db.session.add(new_message)
            db.session.commit()
    else:
        message_text = "Invalid hand gesture"

    return render_template('handgesture.html', prediction=message_text)

@app.route('/')
def home():
    return render_template('home.html')


@app.route('/handgesture')
def handgesture():
    return render_template('handgesture.html')

@app.route('/customize', methods=['GET'])
def customize():
    gestures = Gesture.query.all()
    return render_template('customize.html', gestures=gestures)

@app.route('/message', methods=['GET'])
def MessageTable():
    messages = Message.query.all()
    return render_template('message.html', messages=messages)

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

@app.route('/updateGestures', methods=['POST'])
def updateGestures():
    gestures = Gesture.query.all()
    for gesture in gestures:
        new_message = request.form.get(f'gesture_{gesture.id}_message')
        if new_message:
            gesture.message = new_message
    db.session.commit()

    return redirect(url_for('customize'))


if __name__ == '__main__':
    with app.app_context():
        db.create_all()
        insert_default_gestures()
 
    app.run(debug=True)
