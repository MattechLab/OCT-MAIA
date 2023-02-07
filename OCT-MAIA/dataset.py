from copy import copy
from typing import List

from skimage import io
from skimage.color import rgb2gray
from skimage.filters import threshold_otsu, median
from skimage.measure import label
from skimage.morphology import selem
from skimage.segmentation import clear_border
from skimage.transform import resize
from numpy import count_nonzero, min, max, pad, histogram
from tqdm import tqdm
from PIL import Image
from shutil import rmtree, copy2

import numpy as np
import math
import os
import pickle
import xml.etree.ElementTree as xmlET

from util import make_odd, odd_round

# see: https://stackoverflow.com/a/25389715
from registration import VisualAngle


DATASET = None


def initialize(config_filename=os.path.join(os.path.dirname(__file__), 'dataset.txt')):
    global DATASET

    config_dict = {}
    with open(config_filename, 'r') as config:
        for line in config:
            parts = line.strip('\n').split('=')
            key = parts[0]
            values = parts[1].split(',')
            if len(values) == 1:
                config_dict[key] = values[0]
            else:
                config_dict[key] = values
    DATASET = config_dict
    return config_dict


GROUPS = ['amd', 'control']
DEVICES = ['maia', 'spectralis']
CONTENTS = {
    'maia': ['bcea', 'fundus', 'prl'],
    'spectralis': ['fundus', 'oct']
}
EYES = ['left', 'right']
PARTS_DETAIL = ['up', 'center', 'down']
PARTS = ['full', 'up', 'center', 'down']


def subjects(group='amd'):
    return DATASET[group]


def subjects_subset(group='amd', subset='train'):
    return DATASET[group + subset.capitalize()]


def all_data_identifiers():
    return [(group, subject_id, subject, eye, part)
            for part in PARTS
            for eye in EYES
            for group in GROUPS
            for subject_id, subject in enumerate(subjects(group))]


def _build_filepath(subject, eye='left', extension='png'):
    return f'{subject}_{eye}.{extension}'


def _build_spectralis_filepath(subject, eye):
    return f'{subject}_{eye}.tif'


def _build_single_filename(subject, device='spectralis', eye='left', maia_jpg=False):
    if device == 'maia':
        extension = 'jpg' if maia_jpg else 'png'
    else:
        extension = 'tif'

    return f'{subject}_{eye}.{extension}'


def _read_fundus_filename(group, subject, eye, part):
    directory = os.path.join(DATASET['dataPath'], group, 'spectralis', 'oct', f'{subject}_{eye}_{part}')
    xml_file = _find_volume_description_xml(directory)
    xml_root = xmlET.parse(os.path.join(directory, xml_file))

    for image_entry in xml_root.iter('Image'):
        image_type = image_entry.find('ImageType/Type').text
        if image_type == 'LOCALIZER':
            exam_url = image_entry.find('ImageData/ExamURL').text
            return _extract_filename_from_exam_url(exam_url)

    return None
            

def image_path(subject_id, group='amd', device='spectralis', content='fundus', eye='left', part='full'):
    assert group in GROUPS, 'Unknown group'
    assert device in DEVICES, 'Unknown device'
    assert content in CONTENTS[device], 'Unknown content'
    assert eye in EYES, 'Unknown eye'
    assert part in PARTS, 'Unknown part'
    
    subject = subjects(group)[subject_id]
    directory = os.path.join(DATASET['dataPath'], group, device, content)
    
    if device == 'spectralis' and content == 'fundus':
        filename = _read_fundus_filename(group, subject, eye, part)
        path = os.path.join(DATASET['dataPath'], group, 'spectralis', 'oct', f'{subject}_{eye}_{part}', filename)
    else:
        filename = _build_single_filename(subject, device, eye)
        path = os.path.join(directory, filename)

    if not os.path.exists(path):
        if device == 'maia':
            filename = _build_single_filename(subject, device, eye, maia_jpg=True)
            path = os.path.join(directory, filename)
            if not os.path.exists(path):
                raise FileNotFoundError()
        else:
            raise FileNotFoundError()

    return path


def image(path, convert_to_grayscale=True, oct_tiff=False):
    img = io.imread(path)
    if oct_tiff:
        return img[:, :, 0]  # all channels are equal, grayscale image

    if convert_to_grayscale and len(img.shape) > 2:
        img = rgb2gray(img)
    return img


def z_score(image: np.ndarray):
    return (image - image.mean()) / image.std()


LABEL_TYPES = {'inner', 'outer'}


def label_path(subject_id, group='amd', eye='left', part='full', type='outer'):
    assert group in GROUPS, 'Unknown group'
    assert eye in EYES, 'Unknown eye'
    assert type in LABEL_TYPES, 'Unknown type'
    assert part in PARTS, 'Unknown part'

    subject = subjects(group)[subject_id]
    return os.path.join(DATASET['labelsPath'], group, f'{subject}_{eye}_{part}_{type}.png')


def _find_volume_description_xml(volume_directory):
    files = os.listdir(volume_directory)
    for file in files:
        if file.split('.')[1] == 'xml':
            return file
    raise FileNotFoundError()


def _extract_filename_from_exam_url(exam_url):
    return exam_url.split('\\')[-1]


class AcquisitionContext:
    def __init__(self, width, height, scale_x, scale_y):
        self.width = width
        self.height = height
        self.scale_x = scale_x
        self.scale_y = scale_y

    def __eq__(self, other):
        if not isinstance(other, AcquisitionContext):
            return False
        return (self.width == other.width) and (self.height == other.height) and \
               (self.scale_x == other.scale_x) and (self.scale_y == other.scale_y)

    def rescale_coordinates(self, coordinates):
        y, x = coordinates
        return (y / self.scale_y), (x / self.scale_x)


class OCTScan:
    def __init__(self, sequence_number, filename, start_pixel_coordinate, end_pixel_coordinates):
        self.sequence_number = sequence_number
        self.filename = filename
        self.start = start_pixel_coordinate
        self.end = end_pixel_coordinates

    def image(self):
        return image(self.filename, oct_tiff=True)

    def __repr__(self):
        fields = copy(self.__dict__)
        return str(fields)


class OCTVolume:
    def __init__(self, acquisition_context, oct_scans):
        self.acquisition_context = acquisition_context
        self.scans = sorted(oct_scans, key=lambda oct_scan: oct_scan.sequence_number)

    def merge(self, other):
        assert self.acquisition_context == other.acquisition_context, 'Different acquisition contexts, cannot merge'
        self.scans += other.scans


def _parse_localizer_acquisition_context(xml_root):
    for image_entry in xml_root.iter("Image"):
        image_type = image_entry.find('ImageType/Type').text
        if image_type == 'LOCALIZER':
            ophthalmic_acquisition_context_xml = image_entry.find('OphthalmicAcquisitionContext')
            width = int(ophthalmic_acquisition_context_xml.find('Width').text)
            height = int(ophthalmic_acquisition_context_xml.find('Height').text)
            scale_x = float(ophthalmic_acquisition_context_xml.find('ScaleX').text)
            scale_y = float(ophthalmic_acquisition_context_xml.find('ScaleY').text)
            return AcquisitionContext(width, height, scale_x, scale_y)

    return None


def _parse_oct_scans(xml_root, acquisition_context: AcquisitionContext, volume_directory):
    oct_scans = []
    for image_entry in xml_root.iter('Image'):
        image_type = image_entry.find('ImageType/Type').text
        if image_type == 'OCT':
            sequence_number = int(image_entry.find('ImageNumber').text)
            filename = _extract_filename_from_exam_url(image_entry.find('ImageData/ExamURL').text)

            context_xml = image_entry.find('OphthalmicAcquisitionContext')
            start_pixel_coordinates, end_pixel_coordinates = _parse_pixel_coordinates(
                acquisition_context, context_xml.find('Start/Coord'), context_xml.find('End/Coord')
            )

            oct_scans.append(OCTScan(sequence_number, os.path.join(volume_directory, filename),
                                     start_pixel_coordinates, end_pixel_coordinates))

    return oct_scans


def _parse_pixel_coordinates(acquisition_context, start_coordinates_xml, end_coordinates_xml):
    '''Rounding to centers of pixels'''
    
    start_y, start_x = acquisition_context.rescale_coordinates(
        (float(start_coordinates_xml.find('Y').text), float(start_coordinates_xml.find('X').text))
    )
    end_y, end_x = acquisition_context.rescale_coordinates(
        (float(end_coordinates_xml.find('Y').text), float(end_coordinates_xml.find('X').text))
    )
    
    # Correcting width to expected OCT B-scan width
    rounded_start_x, rounded_end_x = round(start_x - 0.5), round(end_x - 0.5)
    oct_width = rounded_end_x - rounded_start_x + 1
    if abs(768 - oct_width) < abs(512 - oct_width):
        expected_oct_width = 768
    else:
        expected_oct_width = 512
    
    difference = expected_oct_width - oct_width
    abs_difference = abs(difference)
    sign = -1 if difference < 0 else 1
    if difference == 0:
        corrected_start_x = rounded_start_x
        corrected_end_x = rounded_end_x
    else:
        start_correction = abs_difference // 2
        end_correction = abs_difference - start_correction
        corrected_start_x = rounded_start_x - sign * start_correction
        corrected_end_x = rounded_end_x + sign * end_correction
    
    return (round(start_y - 0.5), corrected_start_x), (round(end_y - 0.5), corrected_end_x)


def _oct_volume_from_directory(subject_id, group='amd', eye='left', part='full'):
    subject = subjects(group)[subject_id]
    volume_directory = os.path.join(DATASET['dataPath'], group, 'spectralis', 'oct', f'{subject}_{eye}_{part}')
    xml_file = _find_volume_description_xml(volume_directory)
    xml_root = xmlET.parse(os.path.join(volume_directory, xml_file)).getroot()

    localizer_acquisition_context = _parse_localizer_acquisition_context(xml_root)
    oct_scans = _parse_oct_scans(xml_root, localizer_acquisition_context, volume_directory)

    return OCTVolume(localizer_acquisition_context, oct_scans)


VOLUME_TYPES = {'dense', 'detail'}


def oct_volume(subject_id, group='amd', eye='left', part='full'):
    assert group in GROUPS, 'Unknown group'
    assert eye in EYES, 'Unknown eye'
    assert part in PARTS, 'Unknown part'

    return _oct_volume_from_directory(subject_id, group, eye, part)


class LabelledOCTScan:
    def __init__(self, group, subject_id, eye, part, oct_scan: OCTScan, left_limit, right_limit, label, full_label):
        self.group = group
        self.subject_id = subject_id
        self.eye = eye
        self.part = part
        self.full_label = full_label
        self.oct_scan = oct_scan
        self.left_limit = left_limit
        self.right_limit = right_limit
        self.label = label

    # https://stackoverflow.com/a/38540861
    def __repr__(self):
        fields = copy(self.__dict__)
        del fields['label']
        return str(fields)

    def image(self):
        return image(self.oct_scan.filename, oct_tiff=True)[:, self.left_limit:(self.right_limit + 1)]

    def width(self):
        return self.right_limit - self.left_limit + 1


def labelled_oct_scans(group, name) -> List[LabelledOCTScan]:
    filename = os.path.join(DATASET['labelsPath'], group, f'{name}.pickle')
    with open(filename, 'rb') as file:
        return pickle.load(file)


CLASSIFICATION = {'fixation', 'fundus'}


class OCTStripe(LabelledOCTScan):
    def __init__(self, labelled_oct: LabelledOCTScan,
                 label, left_limit, right_limit, top_limit, bottom_limit):
        super().__init__(labelled_oct.group, labelled_oct.subject_id, labelled_oct.eye, labelled_oct.volume_type,
                         labelled_oct.part, labelled_oct.oct_scan, None, None, None)
        self.label = label
        self.left_limit = left_limit
        self.right_limit = right_limit
        self.top_limit = top_limit
        self.bottom_limit = bottom_limit
        self.roi_height = bottom_limit - top_limit

    def image(self):
        full_image = image(self.oct_scan.filename, oct_tiff=True)
        zero_top_pad, zero_bottom_pad = 0, 0

        # Additional zero padding if computed limits exceed original image
        if self.top_limit < 0:
            zero_top_pad = abs(self.top_limit)
            self.top_limit = 0
        if self.bottom_limit >= full_image.shape[0]:
            zero_bottom_pad = self.bottom_limit - full_image.shape[0]
        padded_full_image = pad(full_image, ((zero_top_pad, zero_bottom_pad), (0, 0)))

        return padded_full_image[self.top_limit:(self.bottom_limit + 1), self.left_limit:(self.right_limit + 1)]


def roi_mask(oct_scan):
    filtered = median(oct_scan, selem.square(3))
    thresholded = filtered > threshold_otsu(filtered)
    filled = _fill_holes(thresholded)
    return largest_object(filled)


def largest_object(binary_image):
    labelled = label(binary_image, connectivity=2)
    hist, _ = histogram(labelled, bins=np.max(labelled))
    biggest_object_label = np.argmax(hist[1:]) + 1
    return labelled == biggest_object_label


def _fill_holes(binary_image):
    return np.logical_or(binary_image, clear_border(~binary_image))


def cut_into_stripes(labelled_oct: LabelledOCTScan, patch_width):
    stripes = []

    # FIXME: should be removed as there should be no OCT scans missing
    try:
        oct_scan = image(labelled_oct.oct_scan.filename, oct_tiff=True)
    except FileNotFoundError:
        return stripes

    mask = roi_mask(oct_scan)
    label_majority = patch_width / 2

    for n in range(labelled_oct.width() // patch_width + 1):
        offset = n * patch_width

        left_limit = labelled_oct.left_limit + offset
        right_limit = left_limit + patch_width

        if left_limit >= 0 and right_limit <= oct_scan.shape[1]:
            mask_stripe = mask[:, left_limit:right_limit]
            # Label is already limited to the used part of the OCT (LabelledOCT.left_limit and right_limit
            # correspond to label's indices 0 and len - 1 respectively)
            labels = labelled_oct.label[offset:(offset + patch_width)]
            aggregated_label = count_nonzero(labels) > label_majority

            top_limit, bottom_limit = mask_vertical_limits(mask_stripe)
            stripes.append(OCTStripe(labelled_oct, aggregated_label, left_limit, right_limit, top_limit, bottom_limit))

    return stripes


def mask_vertical_limits(mask):
    nonzero_vertical_indices = np.nonzero(mask)[0]
    return np.min(nonzero_vertical_indices), np.max(nonzero_vertical_indices)


DETAIL_VA = VisualAngle(15, 768)
DENSE_VA = VisualAngle(20, 512)


def generate_stripe_dataset(patch_angle_width, labelled_data_name):
    patch_widths = {
        'detail': odd_round(DETAIL_VA.to_length(patch_angle_width)),
        'dense': round(DENSE_VA.to_length(patch_angle_width))
    }

    stripes = []
    for group in GROUPS:
        print(group + ':')
        for oct_scan in tqdm(labelled_oct_scans(group, labelled_data_name)):
            stripes += cut_into_stripes(oct_scan, patch_widths[oct_scan.volume_type])

    return stripes


def crop_oct_stripes_vertically(oct_stripes):
    roi_height = make_odd(_minimal_roi_height(oct_stripes))
    for oct_stripe in tqdm(oct_stripes):
        total_padding = roi_height - oct_stripe.roi_height
        top_padding = total_padding // 2
        bottom_padding = total_padding - top_padding
        oct_stripe.top_limit -= top_padding
        oct_stripe.bottom_limit += bottom_padding
    return oct_stripes


def _minimal_roi_height(oct_stripes):
    return max(list(map(lambda oct_stripe: oct_stripe.roi_height, oct_stripes)))


def save_samples(samples, patch_angle_width):
    patch_width = odd_round(DETAIL_VA.to_length(patch_angle_width))

    for id, sample in enumerate(tqdm(samples)):
        sample_image = sample.image()

        # Scale dense volume samples
        if sample.volume_type == 'dense':
            sample_image = _scale_width_to_target(sample_image, patch_width)

        if min(sample_image) < 0 or max(sample_image) > 255:
            raise ValueError('Sample values out of [0, 255] range')

        classification = 'fixation' if sample.label else 'fundus'
        directory = os.path.join(DATASET['stripeDataPath'], classification, sample.group,
                                 str(sample.subject_id), sample.eye, sample.part)
        os.makedirs(directory, exist_ok=True)
        n_files = len(os.listdir(directory))  # https://stackoverflow.com/a/2632251
        file_path = os.path.join(directory, str(n_files) + '.png')
        # https://gist.github.com/ax3l/5781ce80b19d7df3f549
        Image.fromarray(sample_image.astype(np.uint8), mode='L').save(file_path, format='PNG', compress_level=0)


# FIXME: assuming that vertical scaling is not necessary due to 1:1 um export setting
def _scale_width_to_target(scaled_image, width):
    return resize(scaled_image, (scaled_image.shape[0], width), order=2, preserve_range=True)


def remove_selected_stripe_dataset_samples():
    for group in GROUPS:
        for subject, eye, part in _exclusion_list(group):
            subject_id = subject_to_id(subject, group)
            for classification in CLASSIFICATION:
                directory_to_delete = os.path.join(DATASET['stripeDataPath'], classification, group,
                                                   subject_id, eye, part)
                response = input(f'Remove directory {directory_to_delete}? [y/n] ')
                if response == 'y':
                    rmtree(directory_to_delete, ignore_errors=True)


def _exclusion_list(group):
    exclusions = []
    for entry in DATASET[f'{group}Exclude']:
        exclusions.append(tuple(entry.split('_')))
    return exclusions


def subject_to_id(subject, group):
    subjects_list = subjects(group)
    for id, matched_subject in enumerate(subjects_list):
        if subject == matched_subject:
            return str(id)
    return str(-1)


def generate_train_test_datasets():
    dataset_directory = DATASET['datasetPath']
    for classification in CLASSIFICATION:
        os.makedirs(os.path.join(dataset_directory, 'train', classification), exist_ok=True)
        os.makedirs(os.path.join(dataset_directory, 'test', classification), exist_ok=True)

    test_subjects = {}
    for group in GROUPS:
        test_subjects[group] = set(DATASET[group + 'Test'])

    for root, dirs, files in os.walk(DATASET['stripeDataPath']):
        if len(files) > 0:
            classification, group, subject_id, eye, part = root.split('/')[-5:]
            subject = subjects(group)[int(subject_id)]
            prefix = f'{group}_{subject}_{eye}_{part}_'
            for file in files:
                source = os.path.join(root, file)

                subset = 'test' if subject in test_subjects[group] else 'train'
                destination = os.path.join(dataset_directory, subset, classification, prefix + file)
                copy2(source, destination)


PATCH_ANGLE_WIDTH = 1  # degree of VA


if __name__ == "__main__":
    print('Generating raw stripes')
    raw_stripes = generate_stripe_dataset(PATCH_ANGLE_WIDTH, 'labelled_octs_v2')
    print('Cropping')
    cropped_stripes = crop_oct_stripes_vertically(raw_stripes)
    print('Saving')
    save_samples(cropped_stripes, PATCH_ANGLE_WIDTH)
    print('Removing excluded samples')
    remove_selected_stripe_dataset_samples()
    print('Generating train and test datasets')
    generate_train_test_datasets()
