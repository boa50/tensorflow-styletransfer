import functools
import os

from matplotlib import gridspec
import matplotlib.pylab as plt
import numpy as np
import tensorflow as tf
import tensorflow_hub as hub

from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession

config = ConfigProto()
config.gpu_options.allow_growth = True
session = InteractiveSession(config=config)


print("TF Version: ", tf.__version__)
print("TF-Hub version: ", hub.__version__)
print("Eager mode enabled: ", tf.executing_eagerly())
print("GPU available: ", tf.test.is_gpu_available())

def crop_center(image):
    shape = image.shape
    new_shape = min(shape[1], shape[2])
    offset_y = max(shape[1] - shape[2], 0) // 2
    offset_x = max(shape[2] - shape[1], 0) // 2
    image = tf.image.crop_to_bounding_box(
        image, offset_y, offset_x, new_shape, new_shape)
    return image

@functools.lru_cache(maxsize=None)
def load_image(image_url, image_size=(256, 256)):
    image_path = tf.keras.utils.get_file(os.path.basename(image_url)[-128:], image_url)
    return normalize_image(image_path, image_size)

def load_image_local(image_path, image_size=(256,256)):
    return normalize_image(image_path, image_size)

def normalize_image(image_path, image_size):
    img = plt.imread(image_path).astype(np.float32)[np.newaxis, ...]
    if img.max() > 1.0:
        img = img / 255.
    if len(img.shape) == 3:
        img = tf.stack([img, img, img], axis=-1)
    img = crop_center(img)
    img = tf.image.resize(img, image_size, preserve_aspect_ratio=True)
    return img

def show_n(images, titles=('',)):
    n = len(images)
    image_sizes = [image.shape[1] for image in images]
    w = (image_sizes[0] * 6) // 320
    plt.figure(figsize=(w * n, w))
    gs = gridspec.GridSpec(1, n, width_ratios=image_sizes)
    for i in range(n):
        plt.subplot(gs[i])
        plt.imshow(images[i][0], aspect='equal')
        plt.axis('off')
        plt.title(titles[i] if len(titles) > i else '')
    plt.savefig('app/saves/img/test.png')

def save_img(img, name='final'):
    plt.imshow(img[0])
    plt.axis('off')
    plt.savefig('app/saves/img/' + name + '.png', bbox_inches='tight',pad_inches = 0)

output_image_size = 384
content_image_size = (output_image_size, output_image_size)
style_image_size = (256, 256)

# content_image_url = 'https://upload.wikimedia.org/wikipedia/commons/thumb/f/fd/Golden_Gate_Bridge_from_Battery_Spencer.jpg/640px-Golden_Gate_Bridge_from_Battery_Spencer.jpg'  # @param {type:"string"}
# style_image_url = 'https://upload.wikimedia.org/wikipedia/commons/0/0a/The_Great_Wave_off_Kanagawa.jpg'  # @param {type:"string"}
# content_image = load_image(content_image_url, content_image_size)
# style_image = load_image(style_image_url, style_image_size)

content_image_path = 'app/saves/img/content/face_frontal.jpg'
style_image_path = 'app/saves/img/style/The_Great_Wave_off_Kanagawa.jpg'
content_image = load_image_local(content_image_path, content_image_size)
style_image = load_image_local(style_image_path, style_image_size)

style_image = tf.nn.avg_pool(style_image, ksize=[3,3], strides=[1,1], padding='SAME')

hub_handle = 'https://tfhub.dev/google/magenta/arbitrary-image-stylization-v1-256/2'
hub_module = hub.load(hub_handle)

outputs = hub_module(tf.constant(content_image), tf.constant(style_image))
stylized_image = outputs[0]

# show_n([content_image, style_image, stylized_image], titles=['Original content image', 'Style image', 'Stylized image'])
save_img(stylized_image, 'kanagawa')