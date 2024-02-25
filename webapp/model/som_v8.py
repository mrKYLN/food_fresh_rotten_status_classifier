#!/usr/bin/env python3

import numpy as np
from minisom import MiniSom
from sklearn.preprocessing import MinMaxScaler
from skimage.io import imread
from skimage.transform import resize
import matplotlib.pyplot as plt
import glob
import os
import joblib
from skimage import color
from skimage.io import imread
from skimage.transform import resize
from skimage.filters import median
from skimage.morphology import disk

def load_images_and_labels(folders):
    images = []
    labels = []
    for folder in folders:
        for filename in glob.glob(os.path.join(folder, '*.jpg')):
            img = imread(filename)
            if img.ndim == 2:  # Eğer görüntü gri tonlamalıysa
                img = color.gray2rgb(img)  # Görüntüyü RGB'ye dönüştür
            elif img.shape[2] == 4:  # Eğer görüntü RGBA formatındaysa (4 kanal)
                img = img[:, :, :3]  # Alfa kanalını atla ve sadece RGB'yi kullan
            img_resized = resize(img, (256, 256), anti_aliasing=True)  # Yeniden boyutlandır
            images.append(img_resized.reshape(-1))  # Düzleştir ve listeye ekle
            labels.append(os.path.basename(folder))  # Klasör adını etiket olarak kullan
    return np.array(images), labels

folders = ['./level_1', './level_2', './level_3']
images, image_labels = load_images_and_labels(folders)

if len(images) == 0:
    raise ValueError("Hiç resim yüklenemedi. Lütfen klasör yollarını kontrol edin.")

# Rastgele sayı üreteci için seed ayarla
np.random.seed(42)
'''
##############################
def apply_median_filter(images):
    filtered_images = []
    for image in images:
        reshaped_image = image.reshape(128, 128, 3)  
        filtered_image = median(reshaped_image, disk(1)) 
        filtered_images.append(filtered_image.reshape(-1))
    return np.array(filtered_images)

images_filtered = apply_median_filter(images)
##############################
'''
# Verileri ölçeklendir
scaler = MinMaxScaler()
images_scaled = scaler.fit_transform(images)

# SOM modelini oluştur ve eğit
som = MiniSom(x=3, y=1, input_len=256*256*3, sigma=0.12510869927276916, learning_rate=0.8251528597237583)
som.random_weights_init(images_scaled)

starting_weights = som.get_weights().copy() ##
#print(starting_weights)
som.train_random(images_scaled, num_iteration=3000)
ending_weights = som.get_weights().copy()  ##
#print(ending_weights)

feature_names = ['level_1', 'level_2', 'level_3' ]
size = 3
plt.figure(figsize=(10, 10))
for i, f in enumerate(feature_names):
    plt.subplot(3, 3, i+1)
    plt.title(f)
    plt.pcolor(starting_weights[:,:,i].T, cmap='coolwarm')
    plt.xticks(np.arange(size+1))
    plt.yticks(np.arange(size+1))
plt.tight_layout()
plt.show()

plt.figure(figsize=(10, 10))
for i, f in enumerate(feature_names):
    plt.subplot(3, 3, i+1)
    plt.title(f)
    plt.pcolor(ending_weights[:,:,i].T, cmap='coolwarm')
    plt.xticks(np.arange(size+1))
    plt.yticks(np.arange(size+1))
plt.tight_layout()
plt.show()

# Modeli ve ölçeklendiriciyi kaydet
joblib.dump(som, 'som_model.pkl')
joblib.dump(scaler, 'scaler.pkl')

# Kazanan nöronlar ve etiketler için eşleştirme yap ve en sık görülen etiketi belirle
win_labels_map = {}
label_frequency = {}
for i, vector in enumerate(images_scaled):
    winner = som.winner(vector)
    win_labels_map.setdefault(winner, []).append(image_labels[i])
    label_frequency[winner] = max(set(win_labels_map[winner]), key=win_labels_map[winner].count)

joblib.dump(win_labels_map, 'win_labels_map.pkl')  # Kazanan nöronlar ve etiketlerin eşleştirilmesi
joblib.dump(label_frequency, 'label_frequency.pkl')  # Kazanan nöronlar için en sık görülen etiket

# Modelin doğruluğunu hesapla
correct = sum([1 for i, vector in enumerate(images_scaled) if label_frequency[som.winner(vector)] == image_labels[i]])
accuracy = correct / len(images_scaled)

print(f'Accuracy: {accuracy:.2f}')


# SOM haritasını ve kazanan nöronların etiketlerini görselleştir
plt.figure(figsize=(10, 10))

# Her bir etiket için kullanılacak şekiller
markers = {'level_1': 'o', 'level_2': '^', 'level_3': 's'} # o: yuvarlak, ^: üçgen, s: kare
colors = {'level_1': 'r', 'level_2': 'g', 'level_3': 'b'} # r: kırmızı, g: yeşil, b: mavi

for position, label in label_frequency.items():
    plt.plot(position[0] + 0.5, position[1] + 0.5, markers[label], markerfacecolor='None', 
             markeredgecolor=colors[label], markersize=12, markeredgewidth=2)

# SOM haritasının boyutlarını elde et
weights_shape = som.get_weights().shape
x_dim, y_dim = weights_shape[0], weights_shape[1]

# Görselleştirme için sınırları ayarla
plt.xlim([0, x_dim])
plt.ylim([0, y_dim])


plt.title('SOM Haritası')
plt.show()
