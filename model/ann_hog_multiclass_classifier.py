
import os
import numpy as np
import matplotlib.pyplot as plt
from skimage.io import imread
from skimage.transform import resize
from skimage.feature import hog
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay

# === Step 1: Simulate dataset with 5 categories ===
from skimage.draw import random_shapes

CATEGORIES = ["Normal", "Polyp", "Low-grade IN", "High-grade IN", "Adenocarcinoma"]

def generate_mock_image(path):
    image, _ = random_shapes((256, 256), max_shapes=5, num_channels=1, intensity_range=(100, 200))
    image = resize(image[:, :, 0], (256, 256), anti_aliasing=True)
    plt.imsave(path, image, cmap='gray')

def create_mock_multi_class_dataset(base_dir):
    for category in CATEGORIES:
        cat_dir = os.path.join(base_dir, category)
        os.makedirs(cat_dir, exist_ok=True)
        for i in range(10):
            generate_mock_image(os.path.join(cat_dir, f'{category}_{i}.png'))

# === Step 2: Load dataset and extract HOG features ===
def load_multi_class_data(data_dir, img_size=(128, 128)):
    X, y = [], []
    for label, cls in enumerate(CATEGORIES):
        cls_folder = os.path.join(data_dir, cls)
        for fname in os.listdir(cls_folder):
            img_path = os.path.join(cls_folder, fname)
            image = imread(img_path, as_gray=True)
            image_resized = resize(image, img_size)
            features = hog(image_resized, pixels_per_cell=(16, 16), cells_per_block=(2, 2), feature_vector=True)
            X.append(features)
            y.append(label)
    return np.array(X), np.array(y)

# === Step 3: Main training and evaluation pipeline ===
def main():
    base_dir = "/Users/yishao/ColHis-IDS_split/train/200"
    #create_mock_multi_class_dataset(base_dir)
    X, y = load_multi_class_data(base_dir)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = MLPClassifier(hidden_layer_sizes=(50,), max_iter=500, learning_rate_init=0.01, random_state=42)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    print("Accuracy:", model.score(X_test, y_test))
    print(classification_report(y_test, y_pred, target_names=CATEGORIES))

    # Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=CATEGORIES)
    disp.plot(xticks_rotation=45)
    plt.title("Confusion Matrix")
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()
