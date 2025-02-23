import tensorflow as tf
import numpy as np
import os
import cv2
from tensorflow.keras.preprocessing import image
import matplotlib.pyplot as plt

model = tf.keras.models.load_model('candy_sorter_model.h5')

class_labels = ['red', 'yellow', 'blue']

test_images_path = './test_images'

if not os.path.exists(test_images_path):
    print(f"Hata: Test resimleri dizini '{test_images_path}' bulunamadı!")
else:
    seen_files = set()  # Zaten görülen dosyaları kaydedeceğiz
    for img_name in os.listdir(test_images_path):
        img_path = os.path.join(test_images_path, img_name)

        # Dosya uzantısının geçerli bir resim olup olmadığını kontrol et
        if img_name.lower().endswith(('.png', '.jpg', '.jpeg')) and img_name not in seen_files:
            seen_files.add(img_name)  # Bu dosyayı bir kez işlediğimizi kaydediyoruz

            img = cv2.imread(img_path)
            img_resized = cv2.resize(img, (150, 150))
            img_array = image.img_to_array(img_resized)
            img_array = np.expand_dims(img_array, axis=0)

            img_array = img_array / 255.0

            predictions = model.predict(img_array)

            predicted_class = class_labels[np.argmax(predictions)]

            plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
            plt.title(f'Tahmin: {predicted_class}')
            plt.axis('off')
            plt.show()
        else:
            print(f"Uyarı: {img_name} bir resim dosyası değil ya da zaten işlendi, geçiliyor...")
