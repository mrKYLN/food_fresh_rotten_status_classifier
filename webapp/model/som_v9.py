
import joblib
import numpy as np
from minisom import MiniSom
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import GroupShuffleSplit
from skimage.io import imread
from skimage.transform import resize
from skimage import color
import matplotlib.pyplot as plt
import glob
import os
import pandas as pd
from sklearn.metrics import confusion_matrix, classification_report

# Function to load images and labels
def load_images_and_labels(folders):
    images = []
    labels = []
    result = []
    id = 1
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
            result.append((id, img_resized.flatten(), os.path.basename(folder)))
            id = id+1
    return pd.DataFrame(result, columns=["id", "image", "label"])

# Load images
folders = ['./level_1', './level_2', './level_3']
main_df = load_images_and_labels(folders)
#test

# Check if images are loaded
if len(main_df) == 0:
    raise ValueError("No images loaded. Please check folder paths.")

# Veri setini eğitim, doğrulama ve test setlerine ayır
X = main_df.copy().drop(columns=['label'])
y = main_df.label

#X_train, X_temp, y_train, y_temp = train_test_split(original_data["image"], original_data["label"], test_size=0.4, random_state=42)
#X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

gs = GroupShuffleSplit(n_splits=2, test_size=None, train_size=.6, random_state=42)
X_train, X_temp = next(gs.split(X, y, groups=X.id))
X_train_data = X.loc[X_train]
y_train = y.loc[X_train]
X_temp_data = X.loc[X_temp]
y_temp = y.loc[X_temp]

X_temp_data = X_temp_data.reset_index(drop=True)
y_temp = y_temp.reset_index(drop=True)
gs2 = GroupShuffleSplit(n_splits=2, test_size=0.5, random_state=42)
X_val, X_test = next(gs2.split(X_temp_data, y_temp, groups=X_temp_data.id))
X_val_data = X_temp_data.loc[X_val]
y_val = y_temp.loc[X_val]
X_test_data = X_temp_data.loc[X_test]
y_test = y_temp.loc[X_test]

# Verileri yeniden şekillendir ve ölçeklendir
X_train_images = np.array([img.reshape(-1) for img in X_train_data['image'].values])
X_val_images = np.array([img.reshape(-1) for img in X_val_data['image'].values])
X_test_images = np.array([img.reshape(-1) for img in X_test_data['image'].values])

scaler = MinMaxScaler()
X_train_scaled = scaler.fit_transform(X_train_images)
X_val_scaled = scaler.transform(X_val_images)
X_test_scaled = scaler.transform(X_test_images)
print('model kümelemesi başlıyor')
# SOM modelini oluştur ve eğit
som = MiniSom(x=3, y=1, input_len=64*64*3, sigma=0.24279876776123493, learning_rate=0.7704873570467585)
#som = MiniSom(x=3, y=1, input_len=64*64*3, sigma=0.12510869927276916, learning_rate=0.8251528597237583)
som.random_weights_init(X_train_scaled)
som.train_random(X_train_scaled, num_iteration=3000)
print('model kümelemesi tamamlandı')

print('model değerlendirmesi başlıyor')

# Eğitim seti üzerinde kazanan nöronları ve ilişkilendirilen etiketleri kullanarak değerlendirme
win_map = som.win_map(X_train_scaled)
label_map = {}  # Kazanan nöron pozisyonlarına göre etiketleri gruplandır

# Her bir kazanan nöron pozisyonu için etiketleri grupla
for x, label in zip(X_train_scaled, y_train):
    winner = som.winner(x)
    if winner in label_map:
        label_map[winner].append(label)
    else:
        label_map[winner] = [label]

correct_predictions = 0
for position, labels in label_map.items():
    most_common_label = max(set(labels), key=labels.count)  # En sık rastlanan etiketi buluyoruz
    correct_count = labels.count(most_common_label)  # Bu etikete ait doğru tahmin sayısını hesapla
    correct_predictions += correct_count

# Val seti üzerinde modeli değerlendir
accuracy_train = correct_predictions / len(X_train_scaled)
print(f"Eğitim Seti Doğruluğu: {accuracy_train:.2f}")

correct_val = 0
for i, vector in enumerate(X_val_scaled):
    winner = som.winner(vector)
    predicted_label = max(set(label_map[winner]), key=label_map[winner].count)  # Kazanan nöron için en sık rastlanan etiketi tahmin et
    if predicted_label == y_val.iloc[i]:
        correct_val += 1

accuracy_val = correct_val / len(X_val_scaled)
print(f"Doğrulama Seti Üzerindeki Performans: {accuracy_val:.2f}")


# Test seti üzerinde modeli değerlendir
correct_test = 0
for i, vector in enumerate(X_test_scaled):
    winner = som.winner(vector)
    # Kazanan nöron için en sık rastlanan etiketi tahmin et
    if winner in label_map:
        predicted_label = max(set(label_map[winner]), key=label_map[winner].count)
        if predicted_label == y_test.iloc[i]:
            correct_test += 1

accuracy_test = correct_test / len(X_test_scaled)
print(f"Test Seti Doğruluğu: {accuracy_test:.2f}")

print('model değerlendirmesi tamamlandı')

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

#confusion_matrix
print('model matrisleri işleniyor')

# Tahminleri ve gerçek etiketleri saklamak için listeler
y_pred_val = []
y_true_val = []

# Doğrulama seti üzerinde tahminler
for i, vector in enumerate(X_val_scaled):
    winner = som.winner(vector)
    if winner in label_map:
        predicted_label = max(set(label_map[winner]), key=label_map[winner].count)
        y_pred_val.append(predicted_label)
        y_true_val.append(y_val.iloc[i])

# Confusion matrix ve classification report
cm_val = confusion_matrix(y_true_val, y_pred_val, labels=['level_1', 'level_2', 'level_3'])
print("Doğrulama Seti Confusion Matrix:")
print(cm_val)
print("Doğrulama Seti Report:")
print(classification_report(y_true_val, y_pred_val, target_names=['level_1', 'level_2', 'level_3']))


# Tahminleri ve gerçek etiketleri saklamak için listeler
y_pred_test = []
y_true_test = []

# Test seti üzerinde tahminler
for i, vector in enumerate(X_test_scaled):
    winner = som.winner(vector)
    if winner in label_map:
        predicted_label = max(set(label_map[winner]), key=label_map[winner].count)
        y_pred_test.append(predicted_label)
        y_true_test.append(y_test.iloc[i])

# Test seti için confusion matrix ve classification report
cm_test = confusion_matrix(y_true_test, y_pred_test, labels=['level_1', 'level_2', 'level_3'])
print("Test Seti Confusion Matrix:")
print(cm_test)
print("Test Seti Report:")
print(classification_report(y_true_test, y_pred_test, target_names=['level_1', 'level_2', 'level_3']))
##########################################################################################################
print('model matrisleri tamamlandı')

import os
from skimage.io import imsave
print('model yanlış etiketler klasöre yazılıyor')

# Yanlış tahmin edilen resimleri kaydetmek için bir klasör yolu tanımlayın
wrong_preds_folder = 'wrong_predictions'
if not os.path.exists(wrong_preds_folder):
    os.makedirs(wrong_preds_folder)

# Tahminleri ve gerçek etiketleri saklamak için listeler
y_pred_val = []
y_true_val = []
y_val_indexes = []

# Doğrulama seti üzerinde tahminler
for i, vector in enumerate(X_val_scaled):
    winner = som.winner(vector)
    if winner in label_map:
        predicted_label = max(set(label_map[winner]), key=label_map[winner].count)
        y_pred_val.append(predicted_label)
        y_true_val.append(y_val.iloc[i])
        y_val_indexes.append(X_val_data.index[i])  # İndeksleri saklayın


def save_incorrect_predictions(X_data, y_true, y_pred, indexes, folder):
    incorrect_indices = [i for i, (true, pred) in enumerate(zip(y_true, y_pred)) if true != pred]

    for i in incorrect_indices:
        index = indexes[i]
        image_array = X_data.iloc[i]['image'].reshape(64, 64, 3)

        # Resmi 0-255 aralığına ölçeklendir ve uint8 türüne dönüştür
        image_array = (255 * image_array).astype(np.uint8)

        incorrect_label = y_pred[i]
        true_label = y_true[i]
        filename = f'incorrect_index_{index}_true_{true_label}_pred_{incorrect_label}.png'
        filepath = os.path.join(folder, filename)

        # Resmi kaydet
        imsave(filepath, image_array)
print('model yanlış etiketler klasöre kaydediliyor')


# Yanlış tahmin edilen resimleri kaydedin
save_incorrect_predictions(X_val_data, y_true_val, y_pred_val, y_val_indexes, wrong_preds_folder)

print('model yanlış etiketler klasöre kaydedildi')
