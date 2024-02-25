import joblib
import numpy as np
from skimage.io import imread
from skimage.transform import resize

# Modeli, ölçeklendiriciyi ve etiket frekans haritasını yükleme
som = joblib.load('som_model.pkl')
scaler = joblib.load('scaler.pkl')
label_frequency = joblib.load('label_frequency.pkl') 

def predict_label(image_path):
    image = imread(image_path)
    image = resize(image, (64, 64), anti_aliasing=True).flatten()
    image_scaled = scaler.transform([image])
    winner = som.winner(image_scaled)
    # Kazanan hücrenin en sık etiketini ver
    predicted_label = label_frequency.get(winner, "undefined")
    return predicted_label

# Test görüntü yolları
image_paths = ['test_image_1.jpg' , 'test_image_2.jpg', 'test_image_3.jpg', 'test_image_4.jpg', 'test_image_5.jpg',
                'test_image_6.jpg','test_image_7.jpg', 'test_image_8.jpg', 'test_image_9.jpg','test_image_10.jpg','test_image_11.jpg']

# Her bir görüntü için tahmin edilen etiketleri yazdır
for image_path in image_paths:
    predicted_label = predict_label(image_path)
    print(f"{image_path}: Tahmin edilen etiket - {predicted_label}")
