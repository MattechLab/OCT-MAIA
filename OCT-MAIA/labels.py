import dataset
import cv2
import os
import numpy as np

from skimage.measure import label
from skimage.transform import warp, resize
from skimage.io import imsave
from tqdm import tqdm

from coregister import read_warping_transform


def extract_labels_from_maia(group, subject_id, eye):
    try:
        ellipses = extract_bcea_ellipses_lab_space(group, subject_id, eye)
        outer_ellipse, inner_ellipse = split_bcea_ellipses(ellipses)
    except EllipseExtractionError:
        try:
            ellipses = extract_bcea_ellipses_rgb_space(group, subject_id, eye)
            outer_ellipse, inner_ellipse = split_bcea_ellipses(ellipses, connectivity=1)
        except EllipseExtractionError as error:
            raise EllipseExtractionError(error.n_labels, f'{subject_id}_{eye}')

    return fill_holes(outer_ellipse), fill_holes(inner_ellipse)


class EllipseExtractionError(Exception):
    def __init__(self, n_labels, identifier=''):
        self.n_labels = n_labels
        self.identifier = identifier


def fill_holes(image):
    inverted_labeled, n_labelled = label(~image, connectivity=2, return_num=True)
    #BUGFIX:(Holes in BCEA contours has to be filled) Mauro
    if n_labelled == 1:
        ksize = 1
        while n_labelled != 2:
            #print('Filling degraded BCEA countour')
            kernel = np.ones((ksize,ksize), np.uint8)  # note this is a squared kernel
            image = cv2.dilate(1*image.astype('uint8'), kernel, iterations=1).astype(bool) #Dilation to fill holes
            inverted_labeled, n_labelled = label(~image, connectivity=2, return_num=True)
            ksize += 1
    #ENDBUGFIX
    object_sizes = compute_object_sizes(inverted_labeled, n_labelled)
    hole_label = np.argmin(object_sizes) + 1
    return np.logical_or(image, (inverted_labeled == hole_label))

def split_bcea_ellipses(ellipses, connectivity=2):
    labels, n_labelled = label(ellipses, connectivity=connectivity, return_num=True)
    if n_labelled < 2:
        raise EllipseExtractionError(n_labelled)

    object_sizes = compute_object_sizes(labels, n_labelled)
    outer_ellipse_label = np.argmax(object_sizes) + 1
    object_sizes[outer_ellipse_label - 1] = 0
    inner_ellipse_label = np.argmax(object_sizes) + 1

    return (labels == outer_ellipse_label), (labels == inner_ellipse_label)


def compute_object_sizes(labels, n_labelled):
    object_sizes = np.zeros((n_labelled,))
    for n_label in range(1, n_labelled + 1):
        object_sizes[n_label - 1] = np.count_nonzero(labels == n_label)
    return object_sizes


def extract_bcea_ellipses_lab_space(group, subject_id, eye):
    image_path = maia_bcea_image_path(group, subject_id, eye)
    bcea = cv2.imread(image_path)

    bcea_lab = cv2.cvtColor(bcea, cv2.COLOR_RGB2Lab)
    ellipses = bcea_lab[:, :, 1] > 140

    return ellipses


def extract_bcea_ellipses_rgb_space(group, subject_id, eye):
    image_path = maia_bcea_image_path(group, subject_id, eye)
    bcea = cv2.imread(image_path)

    tolerance = 0.3
    red = np.logical_and((1 - tolerance) * 128 <= bcea[:, :, 0], bcea[:, :, 0] <= (1 + tolerance) * 128)
    green = bcea[:, :, 1] < tolerance * 255
    blue = np.logical_and((1 - tolerance) * 128 <= bcea[:, :, 2], bcea[:, :, 2] <= (1 + tolerance) * 128)
    ellipses = np.logical_and(red, green, blue)

    return ellipses


def maia_bcea_image_path(group, subject_id, eye):
    image_path = dataset.image_path(subject_id, group, device='maia', content='bcea', eye=eye)
    if not os.path.exists(image_path):
        raise FileNotFoundError()
    return image_path


def save(image, label_type, group, subject, eye, part):
    path = os.path.join(dataset.DATASET['labelsPath'], group, f'{subject}_{eye}_{part}_{label_type}.png')
    imsave(path, 255 * image.astype(np.uint8), check_contrast=False)


def pad_equally_around(img, target_size):
    total_pad = target_size - img.shape[0]
    leading_pad = total_pad // 2
    trailing_pad = total_pad - leading_pad
    return np.pad(img, ((leading_pad, trailing_pad), (leading_pad, trailing_pad)))


def compute_bcea_areas_measure(measure, group='amd', eye='left', label_type='outer',
                               return_bcea_images=False, exclude={}):
    assert measure in {'intersection', 'union'}, 'Unknown measure specified'

    directory = os.path.join(dataset.DATASET['labelsPath'], group)

    def is_excluded(subject_id, subject_eye, volume):
        return (subject_id, None, None) in exclude or (subject_id, subject_eye, None) in exclude or \
               (subject_id, subject_eye, volume) in exclude

    def belongs_to_labels_of_interest(img_filename):
        extension_dot = img_filename.rfind('.')
        extension = img_filename[extension_dot + 1:]
        if extension == 'png':
            lbl_subject, lbl_eye, lbl_part, lbl_type = img_filename[:extension_dot].split('_')
            subject_id = int(dataset.subject_to_id(lbl_subject, group))
            return not is_excluded(subject_id, lbl_eye, lbl_part) and lbl_eye == eye and lbl_type == label_type
        else:
            return False

    def bcea_label_image_filenames():
        return list(filter(lambda img_filename: belongs_to_labels_of_interest(img_filename), os.listdir(directory)))

    def load_label_image(img_filename):
        lbl_image = cv2.imread(os.path.join(directory, img_filename), cv2.IMREAD_GRAYSCALE)
        # Dense (full) scan images have smaller resolution than detail (up/centre/down)
        if lbl_image.shape[0] != 1536:
            lbl_image = resize(lbl_image, (1536,) * 2)
        return lbl_image

    def crop_image_to_bcea_ellipse(img):
        label_indices = np.argwhere(img > 0)
        y_min, y_max = label_indices[:, 0].min(), label_indices[:, 0].max()
        x_min, x_max = label_indices[:, 1].min(), label_indices[:, 1].max()
        height, width = y_max - y_min + 1, x_max - x_min + 1
        side_len = max(height, width)
        if height > width:
            total_pad = side_len - width
            x_pad = total_pad // 2
            return image[y_min:y_max + 1, x_min - x_pad:x_max + (total_pad - x_pad) + 1]
        else:
            total_pad = side_len - height
            y_pad = total_pad // 2
            return image[y_min - y_pad:y_max + (total_pad - y_pad) + 1, x_min:x_max + 1]

    filenames = bcea_label_image_filenames()
    images = []
    for filename in filenames:
        image = load_label_image(filename)
        images.append(crop_image_to_bcea_ellipse(image))

    largest_size = np.max(list(map(lambda img: img.shape[0], images)))
    if measure == 'intersection':
        measure_output = np.ones((largest_size,) * 2)
    else:  # union
        measure_output = np.zeros((largest_size,) * 2)

    for i in range(len(images)):
        images[i] = pad_equally_around(images[i], largest_size)
        if measure == 'intersection':
            measure_output = np.logical_and(measure_output, images[i])
        else:  # union
            measure_output = np.logical_or(measure_output, images[i])

    if return_bcea_images:
        return measure_output, images
    else:
        return measure_output


NEAREST_NEIGHBOR = 0


def run():
    for group in dataset.GROUPS:
        print(group, 'group:')
        for subject_id, subject in tqdm(enumerate(dataset.subjects(group))):
            for eye in dataset.EYES:
                # Prepare labels from MAIA BCEA
                try:
                    outer_label, inner_label = extract_labels_from_maia(group, subject_id, eye)
                except FileNotFoundError:
                    continue
                except EllipseExtractionError as error:
                    print('Extraction failed:', error.identifier, 'labelled:', error.n_labels)

                for part in dataset.PARTS:
                    # Warp labels
                    try:
                        oct_shape = dataset.image(dataset.image_path(subject_id, group, eye=eye, part=part)).shape
                    except FileNotFoundError:
                        continue

                    warping_transform = read_warping_transform(group, subject, eye, part)
                    warped_outer_label = warp(outer_label, warping_transform,
                                              order=NEAREST_NEIGHBOR, output_shape=oct_shape, preserve_range=True)
                    warped_inner_label = warp(inner_label, warping_transform,
                                              order=NEAREST_NEIGHBOR, output_shape=oct_shape, preserve_range=True)

                    # Save results
                    save(warped_outer_label, 'outer', group, subject, eye, part)
                    save(warped_inner_label, 'inner', group, subject, eye, part)


if __name__ == "__main__":
    run()
