import random

import numpy
from pandas import np
from scipy.ndimage import rotate
from scipy.misc import face
from matplotlib import pyplot as plt

from build_dataset import load_data, save_to_bin_file, show_image_and_label


def get_color(image):
    colors = [image[-1, -1, 0], image[0, 0, 0], image[0, -1, 0], image[-1, 0, 0]]
    color = sum(colors) // 4
    return color


def get_degrees(degrees):
    return random.uniform(10, degrees)


class Augmentation:
    def __init__(self, bin_data_file, bin_labels_file):
        self.data = load_data(bin_data_file)
        self.labels = np.fromfile(bin_labels_file, dtype='uint8')

    def rotate_data(self, degrees):

        rotates = []
        for image in self.data:
            rot = rotate(image, get_degrees(degrees), reshape=False, cval=get_color(image))
            rotates.append(rot)

        return rotates

    def mirror_data(self):
        mirrors = [image[:, ::-1, :] for image in self.data]
        return mirrors

    def save(self, data, label, dir_):
        images = np.array(data)

        save_to_bin_file(images, f"data_dir/{dir_}/data.bin")
        save_to_bin_file(label, f"data_dir/{dir_}/label.bin")

    def merge_files(self, file1_path, file2_path, dir_):
        data1 = load_data(f'{file1_path}/data.bin')
        data2 = load_data(f'{file2_path}/data.bin')
        label1 = np.fromfile(f'{file1_path}/label.bin', dtype='uint8')
        label2 = np.fromfile(f'{file2_path}/label.bin', dtype='uint8')
        data = numpy.concatenate([data1, data2])
        labels = numpy.concatenate([label1, label2])

        self.save(data, labels, dir_)

    def display(self, path):
        """ Make a function to display data from your dataset with the
        corresponding label"""
        labels = np.fromfile(f'{path}/label.bin', dtype='uint8')
        data = load_data(f'{path}/data.bin')
        for img, label in zip(data, labels):
            title = 'yes TFL' if label % 2 else 'no TFL'
            self.show_image(img, title)

    def zoom_data(self, zoom):

        from scipy import ndimage
        import matplotlib.pyplot as plt
        fig = plt.figure()
        ax1 = fig.add_subplot(121)  # left side
        ax2 = fig.add_subplot(122)  # right side
        ascent = self.data[0]
        result = ndimage.zoom(ascent, 3)
        ax1.imshow(ascent)
        ax2.imshow(result)
        plt.show()


def augmentations():
    augmentation = Augmentation('data_dir/train/data.bin', 'data_dir/train/label.bin')
    rotate = augmentation.rotate_data(37)
    mirror = augmentation.mirror_data()
    augmentation.save(rotate, augmentation.labels, "fit_rotate")
    show_image_and_label("rotate1")
    augmentation.save(mirror, augmentation.labels, "mirror")
    augmentation.merge_files('data_dir/d_fit_rotate', "data_dir/mirror", "d_m_r")
    show_image_and_label("d_m_r")

