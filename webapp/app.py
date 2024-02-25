from flask import Flask, render_template, request
from keras.models import load_model
from keras.preprocessing import image
#from model.som_run import predict_label
import joblib
import numpy as np
from skimage.io import imread
from skimage.transform import resize
import os

app = Flask(__name__)


# Modeli, ölçeklendiriciyi ve etiket frekans haritasını yükleme
som = joblib.load("C:\\Users\\mertk\\OneDrive\\Desktop\\food_fresh_rotten_status_classifier_v1\\food_fresh_rotten_status_classifier\\webapp\\model\\som_model.pkl")
scaler = joblib.load("C:\\Users\\mertk\\OneDrive\\Desktop\\food_fresh_rotten_status_classifier_v1\\food_fresh_rotten_status_classifier\\webapp\\model\\scaler.pkl")
label_frequency = joblib.load("C:\\Users\\mertk\\OneDrive\\Desktop\\food_fresh_rotten_status_classifier_v1\\food_fresh_rotten_status_classifier\\webapp\\model\\label_frequency.pkl")  # Doğru dosyayı yükle

def predict_label(image_path):
    image = imread(image_path)
    image = resize(image, (64, 64), anti_aliasing=True).flatten()
    image_scaled = scaler.transform([image])
    winner = som.winner(image_scaled)
    # Kazanan hücrenin en sık etiketini ver
    predicted_label = label_frequency.get(winner, "undefined")
    return predicted_label



"""
def load_project_model():
	with open('../model/som_model.pkl', 'rb') as f:
		model = pickle.load(f)
#model.make_predict_function()
"""
"""
def predict_label(img_path):
	i = image.load_img(img_path, target_size=(100,100))
	i = image.img_to_array(i)/255.0
	i = i.reshape(1, 100,100,3)
	model = load_project_model()
	p = model.predict(i)
	return dic[p[0]]
	return "Rotten"
"""

# routes
@app.route("/", methods=['GET', 'POST'])
def main():
	return render_template("index.html")

@app.route("/about")
def about_page():
	return render_template("about.html")


@app.route("/submit", methods = ['GET', 'POST'])
def get_output():
	if request.method == 'POST':
		img = request.files['my_image']

		directory_path = str(os.getcwd())
		img_path = 'C:\\Users\\mertk\\OneDrive\\Desktop\\food_fresh_rotten_status_classifier_v1\\food_fresh_rotten_status_classifier\\webapp\\static\\images\\' + img.filename
		img.save(img_path)

		p = predict_label(img_path)

	return render_template("index.html", prediction = p, img_path = img.filename)

@app.route("/train")
def train_page():
	return render_template("train.html")

if __name__ =='__main__':
	app.debug = True
	app.run(debug = True)
