import os
import tensorflow as tf
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import numpy as np

def load_images_from_folder(folder_path, image_size=(128, 128)):
    images = []
    labels = []
    class_names = os.listdir(folder_path)
    for label, class_name in enumerate(class_names):
        class_folder = os.path.join(folder_path, class_name)
        if not os.path.isdir(class_folder):
            print(f"Skipping non-directory class folder: {class_folder}")
            continue
        num_images = 0
        for filename in os.listdir(class_folder):
            if filename.endswith('.jpg') or filename.endswith('.png'):
                img_path = os.path.join(class_folder, filename)
                img = load_img(img_path, target_size=image_size)
                img_array = img_to_array(img)
                images.append(img_array)
                labels.append(label)
                num_images += 1
        print(f"Loaded {num_images} images from class '{class_name}'")
    return images, labels, class_names




def make_pairs(images, labels):
    pair_images = []
    pair_labels = []
    num_classes = len(np.unique(labels))
    class_indices = [np.where(np.array(labels) == i)[0] for i in range(num_classes)]

    for idx1 in range(len(images)):
        current_image = images[idx1]
        label = labels[idx1]

        # Check if there are images for the current label
        if len(class_indices[label]) == 0:
            print(f"No images for label {label}")
            continue

        # Positive pair (same class)
        idx2 = np.random.choice(class_indices[label])
        pos_image = images[idx2]

        pair_images.append([current_image, pos_image])
        pair_labels.append(1)

        # Ensure there are other classes with images
        other_class_indices = list(set(range(num_classes)) - set([label]))
        other_class_indices = [i for i in other_class_indices if len(class_indices[i]) > 0]
        if len(other_class_indices) == 0:
            print(f"No other classes with images for label {label}")
            continue

        # Negative pair (different class)
        neg_label = np.random.choice(other_class_indices)
        idx2 = np.random.choice(class_indices[neg_label])
        neg_image = images[idx2]

        pair_images.append([current_image, neg_image])
        pair_labels.append(0)

    return np.array(pair_images), np.array(pair_labels)


def create_siamese_dataset(folder_path, image_size=(128, 128), batch_size=32):
    images, labels, class_names = load_images_from_folder(folder_path, image_size)
    print(f"Total images loaded: {len(images)}")
    print(f"Class names: {class_names}")
    
    pair_images, pair_labels = make_pairs(images, labels)
    print(f"Total pairs generated: {len(pair_images)}")

    if len(pair_images) == 0:
        raise ValueError("No pairs generated. Please check your dataset and class structure.")

    def preprocess(image1, image2, label):
        image1 = tf.image.convert_image_dtype(image1, tf.float32)
        image2 = tf.image.convert_image_dtype(image2, tf.float32)
        return (image1, image2), label

    dataset = tf.data.Dataset.from_tensor_slices((pair_images, pair_labels))
    dataset = dataset.map(lambda x, y: (x[0], x[1], y))
    dataset = dataset.map(preprocess)
    dataset = dataset.shuffle(buffer_size=1024)
    dataset = dataset.batch(batch_size)
    dataset = dataset.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
    
    return dataset
