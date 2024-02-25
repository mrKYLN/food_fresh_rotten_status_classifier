import numpy as np
import os
import glob
from skimage.io import imread
from skimage.transform import resize
from skimage import color
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from minisom import MiniSom
import matplotlib.pyplot as plt
import joblib

def load_images_and_labels(folders):
    images = []
    labels = []
    for folder in folders:
        for filename in glob.glob(os.path.join(folder, '*.jpg')):
            img = imread(filename)
            if img.ndim == 2:
                img = color.gray2rgb(img)
            elif img.shape[2] == 4:
                img = img[:, :, :3]
            img_resized = resize(img, (64, 64), anti_aliasing=True)
            images.append(img_resized.reshape(-1))
            labels.append(os.path.basename(folder))
    return np.array(images), np.array(labels)

folders = ['./level_1', './level_2', './level_3']
images, image_labels = load_images_and_labels(folders)

# Veri setini eğitim, doğrulama ve test setlerine ayır
X_train, X_temp, y_train, y_temp = train_test_split(images, image_labels, test_size=0.4, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

# Verileri ölçeklendir
scaler = MinMaxScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)
X_test_scaled = scaler.transform(X_test)

# SOM modelini oluştur ve eğit
som = MiniSom(x=3, y=1, input_len=64*64*3,  sigma=0.12510869927276916, learning_rate=0.8251528597237583)
som.random_weights_init(X_train_scaled)
som.train_random(X_train_scaled, num_iteration=3000)

###################
# Eğitim seti üzerinde kazanan nöronları ve ilişkilendirilen etiketleri kullanarak değerlendirme
win_map = som.win_map(X_train_scaled)
correct_predictions = 0
for position in win_map:
    predicted_labels = [label for label in win_map[position]]
    actual_labels = [y_train[i] for i, x in enumerate(X_train_scaled) if som.winner(x) == position]
    correct_predictions += sum(1 for i in range(len(predicted_labels)) if predicted_labels[i] == actual_labels[i])
accuracy_train = correct_predictions / len(X_train_scaled)
print(f"Eğitim Seti Doğruluğu: {accuracy_train:.2f}")

# Doğrulama seti için kazanan nöronları belirle ve etiket frekanslarını hesaplama
label_frequency_val = {}  # Sözlüğü doğru bir şekilde başlat
for i, vector in enumerate(X_val_scaled):
    winner = som.winner(vector)
    if winner not in label_frequency_val:
        label_frequency_val[winner] = []  # Kazanan için boş liste oluştur eğer daha önce eklenmediyse
    label_frequency_val[winner].append(y_val[i])

# Her kazanan nöron için en sık görülen etiketi belirle
most_common_labels_val = {}
for winner, labels in label_frequency_val.items():
    most_common_labels_val[winner] = max(set(labels), key=labels.count)

# Doğrulama seti üzerinde modeli değerlendir
correct_val = sum(1 for i, vector in enumerate(X_val_scaled) if most_common_labels_val[som.winner(vector)] == y_val[i])
accuracy_val = correct_val / len(X_val_scaled)
print(f"Doğrulama Seti Üzerindeki Performans: {accuracy_val:.2f}")

# Test seti üzerinde model değerlendirmesi için kod düzenlemesi
correct_test = 0
for i, vector in enumerate(X_test_scaled):
    winner = som.winner(vector)
    predicted_label = label_frequency_test[winner]
    actual_label = y_test[i]
    if predicted_label == actual_label:
        correct_test += 1
accuracy_test = correct_test / len(X_test_scaled)
print(f"Test Seti Doğruluğu: {accuracy_test:.2f}")


# ilk klasor içerisindeki foto patternleri üzerinden accuracy
correct_val = sum([1 for i, vector in enumerate(X_val_scaled) if label_frequency_val[som.winner(vector)] == y_val[i]])
accuracy_val = correct_val / len(X_val_scaled)
print(f"Doğrulama Seti Üzerindeki Performans: {accuracy_val:.2f}")

# Model ve ölçeklendiriciyi kaydet
joblib.dump(som, 'som_model_v2.pkl')  # SOM modelini kaydet
joblib.dump(scaler, 'scaler_v2.pkl')  # Ölçeklendiriciyi kaydet


# Kazanan nöronlar ve etiketler için eşleştirme yap ve test seti için en sık görülen etiketi belirle
win_labels_map_test = {}
label_frequency_test = {}
for i, vector in enumerate(X_test_scaled):
    winner = som.winner(vector)
    win_labels_map_test.setdefault(winner, []).append(y_test[i])
    label_frequency_test[winner] = max(set(win_labels_map_test[winner]), key=win_labels_map_test[winner].count)

# SOM haritasını ve kazanan nöronların etiketlerini görselleştir (Opsiyonel)
plt.figure(figsize=(10, 10))
markers = {'level_1': 'o', 'level_2': '^', 'level_3': 's'}  # o: yuvarlak, ^: üçgen, s: kare
colors = {'level_1': 'r', 'level_2': 'g', 'level_3': 'b'}  # r: kırmızı, g: yeşil, b: mavi
for position, label in label_frequency_test.items():
    plt.plot(position[0] + 0.5, position[1] + 0.5, markers[label], markerfacecolor='None',
             markeredgecolor=colors[label], markersize=12, markeredgewidth=2)
plt.xlim([0, som.x])
plt.ylim([0, som.y])
plt.title('Test Seti Üzerinde SOM Haritası')
plt.show()
