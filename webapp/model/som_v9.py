import numpy as np
from minisom import MiniSom
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from skimage.io import imread
from skimage.transform import resize
from skimage import color
import matplotlib.pyplot as plt
import glob
import os
import joblib
import uuid
import pandas as pd

# Function to load images and labels
def load_images_and_labels(folders):
    images = []
    labels = []
    result = {}
    for folder in folders:
        for filename in glob.glob(os.path.join(folder, '*.jpg')):
            img = imread(filename)
            if img.ndim == 2:  # If the image is grayscale
                img = color.gray2rgb(img)  # Convert to RGB
            elif img.shape[2] == 4:  # If the image is RGBA
                img = img[:, :, :3]  # Skip the alpha channel
            img_resized = resize(img, (64, 64), anti_aliasing=True)  # Resize image
            #images.append(img_resized.flatten())  # Flatten and append to list
            #labels.append(os.path.basename(folder))  # Use folder name as label
            result["uuid"] = uuid.uuid4()
            result["image"] = img_resized.flatten()
            result["label"] = os.path.basename(folder)


    return result

# Load images
folders = ['./level_1', './level_2', './level_3']
image_and_labels_dict = load_images_and_labels(folders)


# Check if images are loaded
if len(image_and_labels_dict) == 0:
    raise ValueError("No images loaded. Please check folder paths.")

# Veri setini eğitim, doğrulama ve test setlerine ayır
original_data = pd.DataFrame(image_and_labels_dict)
X_train, X_temp, y_train, y_temp = train_test_split(original_data["image"], original_data["label"], test_size=0.4, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

# Verileri ölçeklendir
scaler = MinMaxScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)
X_test_scaled = scaler.transform(X_test)

# SOM modelini oluştur ve eğit
som = MiniSom(x=3, y=1, input_len=64*64*3, sigma=0.12510869927276916, learning_rate=0.8251528597237583)
som.random_weights_init(X_train_scaled)
som.train_random(X_train_scaled, num_iteration=3000)

# Eğitim seti üzerinde kazanan nöronları ve ilişkilendirilen etiketleri kullanarak değerlendirme
win_map = som.win_map(X_train_scaled)
label_map = som.labels_map(X_train_scaled,y_train)
correct_predictions = 0

for label_coord, position in win_map.items():
    # Gerçek etiketleri bu pozisyondaki X_train örneklerine dayanarak al
    predicted_labels = [y_train[i] for i, x in enumerate(X_train_scaled) if som.winner(x) == label_coord]
    
    # predicted_labels bir numpy dizisiyse listeye dönüştür
    #if isinstance(position, np.ndarray):
    #    position = position.tolist()
    
    # actual_labels bir numpy dizisiyse listeye dönüştür
    #if isinstance(predicted_labels, np.ndarray):
    #    predicted_labels = predicted_labels.tolist()

    # Her bir tahmin edilen etiket için, gerçek etiketler listesinde olup olmadığını kontrol et
    for predicted_label in predicted_labels:
       if predicted_label in y_train:
            correct_predictions += 1

print(position.head(10))
print(predicted_labels.head(10))
# Doğruluğu hesapla ve yazdır
accuracy_train = correct_predictions / len(X_train_scaled)
print(f"Eğitim Seti Doğruluğu: {accuracy_train:.2f}")


# Doğrulama seti üzerinde modeli değerlendir
correct_val = sum(1 for i, vector in enumerate(X_val_scaled) if y_val[i] in win_map[som.winner(vector)])
accuracy_val = correct_val / len(X_val_scaled)
print(f"Doğrulama Seti Üzerindeki Performans: {accuracy_val:.2f}")

# Test seti üzerinde modeli değerlendir
correct_test = sum(1 for i, vector in enumerate(X_test_scaled) if y_test[i] in win_map[som.winner(vector)])
accuracy_test = correct_test / len(X_test_scaled)
print(f"Test Seti Doğruluğu: {accuracy_test:.2f}")

# Model ve ölçeklendiriciyi kaydet
joblib.dump(som, 'som_model.pkl')
joblib.dump(scaler, 'scaler.pkl')

# Results
print(f"Training Set Accuracy: {accuracy_train:.2f}")
print(f"Validation Set Accuracy: {accuracy_val:.2f}")
print(f"Test Set Accuracy: {accuracy_test:.2f}")

# Plot SOM weight maps
plt.figure(figsize=(10, 10))
for i, f in enumerate(win_map):
    plt.subplot(3, 3, i+1)
    plt.title(f'Node: {f}')
    plt.imshow(som.get_weights()[f[0], f[1], :].reshape(64, 64, 3), interpolation='none')
plt.tight_layout()
plt.show()

