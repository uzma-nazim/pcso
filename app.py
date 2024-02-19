from flask import Flask, render_template, request, jsonify
from keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
import webbrowser

app = Flask(__name__)
model = load_model("bestmodel.h5")

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        # Get the file from the request
        file = request.files['file']

        # Save the file to the static/images folder
        img_path = f'static/images/{file.filename}'
        file.save(img_path)

        # Make a prediction using your model
        pred = predict_image(img_path)

        # Render the result page with the prediction
        return render_template('result.html', prediction=pred, img_path=img_path)

def predict_image(img_path):
    img = image.load_img(img_path, target_size=(224, 224))
    img_array = image.img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    pred = model.predict(img_array)

    if pred[0][0] > 0.5:
        webbrowser.open_new_tab('https://fyp-blue.vercel.app/success')
        return 'Not Affected'
    else:
        webbrowser.open_new_tab('https://example.com/affected')
        return 'Affected'

if __name__ == '__main__':
    app.run(debug=True)
