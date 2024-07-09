import os
import random
import tensorflow as tf
from keras.preprocessing.image import load_img, img_to_array


def custom_transform(image):
    image = tf.image.resize(image, (224, 224))
    image = image / 255.0
    return image


class SiameseDataset(tf.data.Dataset):
    def __new__(cls, data_dir, transform=None, batch_size=4, buffer_size=1000):
        """
        Args:
            data_dir (str): Path to the directory with the dataset.
            transform (callable, optional): Optional transform to be applied on a sample.
            batch_size (int): Number of samples per batch.
            buffer_size (int): Buffer size for shuffling.
        """
        self = super(SiameseDataset, cls).__new__(cls)

        self.data_dir = data_dir
        self.transform = transform

        # Assume data_dir contains subdirectories for each class
        self.classes = os.listdir(data_dir)
        self.image_paths = {cls: [os.path.join(data_dir, cls, img) for img in os.listdir(os.path.join(data_dir, cls))] for cls in self.classes}
        self.all_image_paths = [img_path for cls in self.image_paths.values() for img_path in cls]

        # Create the dataset
        dataset = tf.data.Dataset.from_tensor_slices(self.all_image_paths)
        dataset = dataset.map(self._parse_function, num_parallel_calls=tf.data.AUTOTUNE)
        dataset = dataset.shuffle(buffer_size).batch(batch_size).prefetch(tf.data.AUTOTUNE)

        return dataset

    def _parse_function(self, anchor_path):
        anchor_class = tf.strings.split(anchor_path, os.path.sep)[-2]

        positive_path = tf.numpy_function(self._get_positive_example, [anchor_class, anchor_path], tf.string)
        negative_path = tf.numpy_function(self._get_negative_example, [anchor_class], tf.string)

        anchor_image = self._load_image(anchor_path)
        positive_image = self._load_image(positive_path)
        negative_image = self._load_image(negative_path)

        return (anchor_image, positive_image, negative_image), anchor_class

    def _get_positive_example(self, anchor_class, anchor_path):
        positive_path = random.choice(self.image_paths[anchor_class.decode('utf-8')])
        while positive_path == anchor_path.decode('utf-8'):
            positive_path = random.choice(self.image_paths[anchor_class.decode('utf-8')])
        return positive_path.encode('utf-8')

    def _get_negative_example(self, anchor_class):
        negative_class = random.choice(self.classes)
        while negative_class == anchor_class.decode('utf-8'):
            negative_class = random.choice(self.classes)
        negative_path = random.choice(self.image_paths[negative_class])
        return negative_path.encode('utf-8')

    def _load_image(self, image_path):
        image = tf.io.read_file(image_path)
        image = tf.image.decode_jpeg(image, channels=3)
        if self.transform:
            image = self.transform(image)
        else:
            image = tf.image.resize(image, (256,256))
            image = image / 255.0
        return image
