import numpy as np
from minisom import MiniSom
from sklearn.preprocessing import MinMaxScaler
from skimage.io import imread
from skimage.transform import resize
import glob
import os
import joblib
from hyperopt import hp, fmin, tpe, Trials

def load_images_and_labels(folders):
    images = []
    labels = []
    for folder in folders:
        for filename in glob.glob(os.path.join(folder, '*.jpg')):
            img = imread(filename)
            if img.ndim == 2:  # Eğer görüntü gri tonlamalıysa
                img = color.gray2rgb(img)  # Görüntüyü RGB'ye dönüştür
            elif img.shape[2] == 4:  # Eğer görüntü RGBA formatındaysa
                img = img[:, :, :3]  # Alfa kanalını atla
            img_resized = resize(img, (64, 64), anti_aliasing=True)
            images.append(img_resized.flatten())
            labels.append(os.path.basename(folder))
    return np.array(images), np.array(labels)

folders = ['level_1', 'level_2', 'level_3']
images, image_labels = load_images_and_labels(folders)

np.random.seed(42)

scaler = MinMaxScaler()
images_scaled = scaler.fit_transform(images)

space = {
    'sigma': hp.uniform('sigma', 0.1, 1.0),
    'learning_rate': hp.uniform('learning_rate', 0.1, 1.0),
    'num_iteration': hp.choice('num_iteration', [100, 500, 1000, 5000])
}

def objective(params):
    sigma = params['sigma']
    learning_rate = params['learning_rate']
    num_iteration = params['num_iteration']
    
    som = MiniSom(x=3, y=1, input_len=64*64*3, sigma=sigma, learning_rate=learning_rate)
    som.random_weights_init(images_scaled)
    som.train_random(images_scaled, num_iteration)
    
    win_map = som.win_map(images_scaled)
    predicted_labels = {}
    for position, vectors in win_map.items():
        labels = [image_labels[np.argwhere(images_scaled == vector)[0][0]] for vector in vectors]
        if labels:
            predicted_labels[position] = max(set(labels), key=labels.count)
    
    correct = sum(1 for i, vector in enumerate(images_scaled) if predicted_labels.get(som.winner(vector), None) == image_labels[i])
    accuracy = correct / len(images_scaled)
    
    return 1 - accuracy

trials = Trials()
best_params = fmin(
    fn=objective,
    space=space,
    algo=tpe.suggest,
    max_evals=100,
    trials=trials
)

print(f'En iyi parametreler: {best_params}')

# En iyi parametreleri ve elde edilen en yüksek doğruluk değerini bir dosyaya kaydet
with open("som_hyperparameter_optimization_results_v1.txt", "w") as f:
    f.write(f"En iyi parametreler: {best_params}\n")
    best_accuracy = 1 - trials.best_trial['result']['loss']
    f.write(f"Elde edilen en yüksek doğruluk: {best_accuracy:.4f}\n")
