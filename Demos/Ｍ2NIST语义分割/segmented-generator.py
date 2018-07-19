import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

data = pd.read_csv('train.csv').as_matrix()

img_shape=(64,84)
dataset_sz = 5000
max_digits_per_image = 3

np.random.seed(1234)


def combine(single_images, max_digits_per_image, canvas_sz):
    # Number (random) of digits to be combined between 1 and max_digits_per_image.
    nb_digits = np.random.randint(low=2, high=max_digits_per_image + 1)
    # Indices (random) of digit images to be combined.
    rand_indices = np.random.randint(0, len(single_images), nb_digits)

    src_images = single_images[rand_indices, 1:]
    src_labels = single_images[rand_indices, 0]

    # Segmented output image once channel per digit.
    labels = np.zeros([*canvas_sz, 10], dtype=single_images.dtype)

    for i in range(nb_digits):
        x_off_start = np.random.randint(i * 28, i * 28 + 10)
        y_off_start = np.random.randint(0, canvas_sz[0] - 28 + 1)

        x_off_end = x_off_start + 28
        y_off_end = y_off_start + 28

        if x_off_end <= canvas_sz[1] and y_off_end <= canvas_sz[0]:
            src_img = src_images[i].reshape([28, 28])
            src_digit = src_labels[i]
            labels[y_off_start:y_off_end, x_off_start:x_off_end, src_digit] = src_img

    canvas = np.max(labels, axis=2)
    labels = np.clip(labels, a_min=0, a_max=1)

    return canvas, labels

combined = []
segmented = []

for i in range(dataset_sz):
    img, segments = combine(data,max_digits_per_image, img_shape)
    combined.append(img)
    segmented.append(segments)

np.save('combined.npy',combined)
np.save('segmented.npy',segmented)

combined = np.load('combined.npy')
segmented = np.load('segmented.npy')

rand_index = np.random.randint(0,len(combined))
c,s = combined[rand_index], segmented[rand_index]

plt.imshow(c, cmap='gray')
plt.show()

for i in range(10):
    #plt.figure(figsize=(0.5,0.5))
    plt.imshow(s[:,:,i],cmap='gray')
    plt.show()

