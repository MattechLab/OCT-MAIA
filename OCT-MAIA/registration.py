import cv2
import numpy as np
import math

import util

from skimage.transform import resize, PiecewiseAffineTransform, warp


# See: https://www.sr-research.com/eye-tracking-blog/background/visual-angle/
class VisualAngle:
    def __init__(self, angle, length):
        self.angle = angle
        self.length = length
        self.distance = length / (2 * math.tan(math.radians(angle / 2)))

    def to_length(self, angle):
        return 2 * self.distance * math.tan(math.radians(angle / 2))

    def to_angle(self, length):
        return math.degrees(2 * math.atan(length / (2 * self.distance)))


def extract_window(image, center, radius):
    assert type(center) == tuple and len(center) == 2, 'Center must be a tuple of window center coordinates'
    if isinstance(radius, int):
        v_rad = h_rad = radius
    elif isinstance(radius, Tuple) and len(radius) == 2:
        v_rad, h_rad = radius
    else:
        raise ValueError('Radius must be an integer or a Tuple')

    x, y = center
    return image[(x - v_rad):(x + v_rad + 1), (y - h_rad):(y + h_rad + 1)]


def normalized_cross_correlation(image, template):
    v_template, h_template = template.shape
    assert v_template % 2 == 1 and h_template % 2 == 1, 'Template dimensions must be odd'

    demeaned_template = template - np.mean(template)
    var_template = np.sum(np.square(demeaned_template))

    v_pad, h_pad = v_template // 2, h_template // 2
    v_image, h_image = image.shape

    ncc = np.full(image.shape, fill_value=-1, dtype=np.float32)
    for x in range(v_pad, v_image - v_pad):
        for y in range(h_pad, h_image - h_pad):
            patch = image[x - v_pad:x + v_pad + 1, y - h_pad:y + h_pad + 1]
            demeaned_patch = patch - np.mean(patch)
            var_patch = np.sum(np.square(demeaned_patch))

            ncc[x, y] = np.sum(np.multiply(demeaned_patch, demeaned_template)) / np.sqrt(var_template * var_patch)

    return ncc


def argmax_image(image):
    index = np.argmax(image.flatten())
    size = image.shape[0]
    x = index // size
    y = index % size
    return x, y


def make_odd(value):
    if value % 2 == 0:
        return value + 1
    else:
        return value


def optic_nerve_patch(image, side='right', vertical_span=0.4, horizontal_span=0.1):
    assert side in ('right', 'left'), "Side must be either 'right' or 'left'"

    side = image.shape[0]
    center = side // 2

    # Calculate odd absolute span (in pixels)
    abs_h_span, abs_v_span = make_odd(round(horizontal_span * side)), make_odd(round(vertical_span * side))
    abs_v_rad = abs_v_span // 2

    if side == 'right':
        return image[center - abs_v_rad:center + abs_v_rad + 1, -abs_h_span:]
    else:
        return image[center - abs_v_rad:center + abs_v_rad + 1, :abs_h_span]


class Tuple:
    def __init__(self, raw_tuple):
        self.raw = raw_tuple

    def is_scalar(self, other):
        if isinstance(other, int) or isinstance(other, float):
            return True
        elif len(self.raw) == len(other):
            return False
        raise ValueError('Operand must be either a scalar of same size tuple')

    def __len__(self):
        return len(self.raw)

    def __add__(self, other):
        if self.is_scalar(other):
            return Tuple((elem + other for elem in self.raw))
        else:
            return Tuple((elem + other_elem for elem, other_elem in zip(self.raw, other)))

    def __sub__(self, other):
        if self.is_scalar(other):
            return Tuple((elem - other for elem in self.raw))
        else:
            return Tuple((elem - other_elem for elem, other_elem in zip(self.raw, other)))

    def __mul__(self, other):
        if self.is_scalar(other):
            return Tuple((elem * other for elem in self.raw))
        else:
            return Tuple((elem * other_elem for elem, other_elem in zip(self.raw, other)))

    def __truediv__(self, other):
        if self.is_scalar(other):
            return Tuple((elem / other for elem in self.raw))
        else:
            return Tuple((elem / other_elem for elem, other_elem in zip(self.raw, other)))

    # see: https://stackoverflow.com/a/40242615
    def __getitem__(self, item):
        if item < len(self.raw):
            return self.raw[item]
        else:
            raise IndexError(f'Index {item} out of tuple bounds')


class Patch:
    def __init__(self, patch, v_span, h_span, image_shape):
        self.patch = patch
        self.v_span = v_span
        self.h_span = h_span

    def shape(self):
        return self.patch.shape

    def source_image_shape(self):
        return self.image.shape


def crop_with_optic_nerve_patch(image, patch_shape, found_patch_center, patch_source_image_side, side='right'):
    v_patch, h_patch = patch_shape
    v_center_patch, h_center_patch = found_patch_center

    source_image_radius = patch_source_image_side // 2
    if v_center_patch < source_image_radius:
        shift = source_image_radius - v_center_patch
        target_side = 2 * v_center_patch + 1
        v_span = (0, target_side)
    else:
        shift = 0
        target_side = patch_source_image_side
        v_span = (v_center_patch - source_image_radius, v_center_patch + source_image_radius + 1)

    if side == 'right':
        right_limit = h_center_patch + h_patch // 2 + 1 - shift
        h_span = (right_limit - target_side + shift, right_limit)
    else:
        left_limit = h_center_patch - h_patch // 2 + shift
        h_span = (left_limit, left_limit + target_side)

    return image[v_span[0]:v_span[1], h_span[0]:h_span[1]], shift


def crop_to_visual_angle(angle, source_image, source_image_va: VisualAngle):
    target_side = round(source_image_va.to_length(angle))
    crop_size = (source_image.shape[0] - target_side) // 2
    return source_image[crop_size:-crop_size, crop_size:-crop_size]


def make_odd_shape(image):
    if image.shape[0] % 2 == 0:
        return image[1:, 1:]
    else:
        return image


def odd_round(value):
    floor = math.floor(value)
    if floor % 2 == 1:
        return floor
    else:
        return floor + 1


def resize_to_same_scale(image1, image1_va: VisualAngle, image2, image2_va: VisualAngle):
    one_degree_length1 = image1_va.to_length(1)
    one_degree_length2 = image2_va.to_length(1)

    if one_degree_length1 > one_degree_length2:
        target_side = odd_round(image2_va.to_length(image1_va.angle))
        return resize(image1, (target_side, target_side)), image2
    elif one_degree_length1 < one_degree_length2:
        target_side = odd_round(image1_va.to_length(image2_va.angle))
        return image1, resize(image2, (target_side, target_side))
    else:
        return image1, image2


def crop_out_shift(image, shift):
    if shift == 0:
        return image
    else:
        return image[shift:-shift, shift:-shift]


# Note: is not properly implemented, cropping may fail and produce unexpected results
def ncc_registration(oct_fundus, maia_fundus, eye='left'):
    oct_fundus = make_odd_shape(oct_fundus)
    maia_fundus = make_odd_shape(maia_fundus)
    oct_va = VisualAngle(30, oct_fundus.shape[0])
    maia_va = VisualAngle(36, maia_fundus.shape[0])

    oct_fundus_resized, maia_fundus_resized = resize_to_same_scale(oct_fundus, oct_va, maia_fundus, maia_va)
    oct_optic_nerve = optic_nerve_patch(oct_fundus_resized, side=eye, horizontal_span=0.2)
    ncc = normalized_cross_correlation(maia_fundus_resized, oct_optic_nerve)
    patch_localisation = argmax_image(ncc)

    maia_aligned, shift = crop_with_optic_nerve_patch(
        maia_fundus_resized, oct_optic_nerve.shape, patch_localisation, oct_fundus_resized.shape[0], side=eye
    )
    oct_aligned = crop_out_shift(oct_fundus_resized, shift)

    return oct_aligned, maia_aligned


def root_sift(descriptors):
    l1_normalized = (descriptors.T / np.linalg.norm(descriptors, ord=1, axis=1)).T
    return np.sqrt(l1_normalized)


KEYPOINTS_EXTRACTION_PARAMETERS = {
    'gridRelativeSize': 0.01,
    'clipLimit': 2,
    'neighborhood': 7,
    'sigmaColor': 150
}


def extract_keypoints_and_descriptors(image):
    # Preprocessing
    grid_size = round(image.shape[0] * KEYPOINTS_EXTRACTION_PARAMETERS['gridRelativeSize'])
    equalized = cv2.createCLAHE(clipLimit=KEYPOINTS_EXTRACTION_PARAMETERS['clipLimit'],
                                tileGridSize=(grid_size,)*2).apply(util.uint8_255(image))
    filtered = cv2.bilateralFilter(equalized, d=KEYPOINTS_EXTRACTION_PARAMETERS['neighborhood'],
                                   sigmaColor=KEYPOINTS_EXTRACTION_PARAMETERS['sigmaColor'],
                                   sigmaSpace=3 * grid_size)

    # RootSIFT
    sift = cv2.SIFT_create()
    keypoints, descriptors = sift.detectAndCompute(filtered, None)
    return keypoints, root_sift(descriptors)


def match_descriptors(query_descriptors, train_descriptors):
    bf_matcher = cv2.BFMatcher_create(cv2.NORM_L2, crossCheck=True)
    return bf_matcher.match(query_descriptors, train_descriptors)


def filter_used_points(query_keypoints, train_keypoints, matches):
    query_points = np.float32([query_keypoints[match.queryIdx].pt for match in matches])
    train_points = np.float32([train_keypoints[match.trainIdx].pt for match in matches])
    _, used_mask = cv2.findHomography(query_points, train_points, cv2.RANSAC)
    used_mask = used_mask.squeeze().astype(bool)
    return query_points[used_mask], train_points[used_mask]


OCT_VA = 30
MAIA_VA = 36


def get_relative_scaling_factor(oct_side_length):
    oct_va = VisualAngle(OCT_VA, oct_side_length)
    maia_scaled_side = round(oct_va.to_length(MAIA_VA))
    return maia_scaled_side / oct_side_length


def create_warping_transformation(target_side, source_points, target_points):
    # Find corners mapping
    homography, _ = cv2.findHomography(source_points, target_points, method=0)
    corners = np.float32([
        [[0, 0]],
        [[0, target_side]],
        [[target_side, target_side]],
        [[target_side, 0]]
    ])
    target_corners = cv2.perspectiveTransform(corners, homography)

    # Create transformation
    warping_transformation = PiecewiseAffineTransform()
    source_points = np.append(source_points, corners.reshape(-1, 2), axis=0)
    target_points = np.append(target_points, target_corners.reshape(-1, 2), axis=0)

    successful = warping_transformation.estimate(source_points, target_points)
    if successful:
        return warping_transformation
    else:
        return None


def warp_and_align(image, warping_transformation, target_side):
    return warp(image, warping_transformation, output_shape=(target_side,)*2)
