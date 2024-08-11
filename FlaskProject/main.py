
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
from flask_migrate import Migrate

app = Flask(__name__)

# Define constants
IMAGE_SIZE = 150

# Load the pre-trained model
model = load_model('model1.h5')

app.config['SQLALCHEMY_DATABASE_URI'] = 'mysql://aap:mysql@mysql-container:3306/aap'
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
    senttime = db.Column(db.DateTime, nullable=False, default=datetime.utcnow)

    def __repr__(self):
        return f'<Message {self.id}: {self.message} at {self.senttime}>'

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



    return jsonify({'result':predicted_class })

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
        output = NaqSentModel(inputs)
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
 
    app.run(debug=True)
