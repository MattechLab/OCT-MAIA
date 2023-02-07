import argparse
import numpy as np
import os
import pickle
import re
import cv2 as cv
import coregister
import dataset
import labels
import pairing
import segmentation
import util

from PIL import Image
from typing import List, Dict, Tuple
from tqdm import tqdm


class Identifier:
    def __init__(self):
        self.tuple = ()

    def to_filename(self):
        filename = ''
        for elem in self.tuple:
            filename += str(elem) + '_'
        return filename[:-1]


class EyeIdentifier(Identifier):
    def __init__(self, group, subject_id, eye):
        """
        Identifies subject's eye

        :param group: subject's group ('amd' or 'control')
        :param subject_id: subject's id (integers starting from 0 within the group)
        :param eye: subject's eye ('left' or 'right')
        """
        super().__init__()
        self.tuple = (group, subject_id, eye)

    @property
    def group(self):
        return self.tuple[0]

    @property
    def subject_id(self):
        return self.tuple[1]

    @property
    def eye(self):
        return self.tuple[2]

    def __getitem__(self, item):
        if item < 3:
            return self.tuple[item]
        else:
            raise IndexError()


class VolumesIdentifier(EyeIdentifier):
    def __init__(self, group, subject_id, eye, volume_type, parts):
        """
        Identifies used OCT volumes for one subject's eye

        :param group: subject's group ('amd' or 'control')
        :param subject_id: subject's id (integers starting from 0 within the group)
        :param eye: subject's eye ('left' or 'right')
        :param volume_type: type of the OCT volumes used: 'dense' - single OCT volume per eye with 49 B-scans
            covering 20x20 visual angles; 'detail' - up to 3 OCT volumes per eye ('up', 'center', 'down'),
            each covering 15x5 visual angles with 49 B-scans. For the OCT volumes details, e.g. visual angle coverage
            and number of B-scans check the Spectralis manual (those above were from the top of my head).
        :param parts: names for all the actual OCT volumes used for this eye (see PartIdentifier). In case of 'dense'
            OCT volume there is only one part - 'full'. For 'detail' volumes there is up to 3 parts: 'up', 'center' and
            'down'.
        """
        super().__init__(group, subject_id, eye)
        self._volume_type = volume_type
        self._parts = parts

    @property
    def volume_type(self):
        return self._volume_type

    @property
    def parts(self):
        return self._parts

    def part_identifiers(self):
        """
        Gets a list of volume part identifiers comprising this volumes identifier.
        """
        return [PartIdentifier(self.group, self.subject_id, self.eye, part) for part in self._parts]


class PartIdentifier(EyeIdentifier):
    def __init__(self, group, subject_id, eye, part):
        """
        Identifies one actual OCT volume

        :param group: subject's group ('amd' or 'control')
        :param subject_id: subject's id (integers starting from 0 within the group)
        :param eye: subject's eye ('left' or 'right')
        :param part: which part of the eye fundus this volume represents. If it comes from a 'dense' scan the only
            part can be 'full'. If it comes from a 'detail' scan it can be either 'up', 'center' or 'down' and
            represent a respective part of the eye fundus.
        """
        super().__init__(group, subject_id, eye)
        self.tuple += (part,)

    def __getitem__(self, item):
        if item < 4:
            return self.tuple[item]
        else:
            raise IndexError()

    @property
    def part(self):
        return self.tuple[3]

    def to_eye_identifier(self):
        """
        Converts it to eye identifier, in essence dropping the volume part specification.
        """
        return EyeIdentifier(self.group, self.subject_id, self.eye)


def all_part_identifiers():
    """
    All possible part identifiers in the dataset. Not all of them exist as subjects could have had only a 'dense' or
    only a 'detail' scan or could have had only one eye scanned.
    """
    return [PartIdentifier(group, subject_id, eye, part)
            for part in dataset.PARTS
            for eye in dataset.EYES
            for group in dataset.GROUPS
            for subject_id, _ in enumerate(dataset.subjects(group))]


def all_eye_identifiers():
    """
    All possible eye identifiers in the dataset. Not all of them exist as subjects could have had only one eye scanned.
    """
    return [EyeIdentifier(group, subject_id, eye)
            for eye in dataset.EYES
            for group in dataset.GROUPS
            for subject_id, _ in enumerate(dataset.subjects(group))]


# Saving and loading intermediate results

def save_pickle(folder, identifier: Identifier, data):
    """
    Save given data as Python's pickle

    :param folder: folder in which to save the given data
    :param identifier: identifier that will be used to create the saved file's filename
    :param data: data to be pickled and saved
    :return: None
    """
    with open(pickle_file_path(folder, identifier), 'wb') as file:
        pickle.dump(data, file)


def load_pickle(folder, identifier: Identifier):
    """
    Loads previously pickled data

    :param folder: folder in which the data was saved
    :param identifier: data identifier with which it was saved, used to reconstruct the filename
    :return: retrieved data or None if given data could not be found
    """
    filepath = pickle_file_path(folder, identifier)
    if os.path.exists(filepath):
        with open(filepath, 'rb') as file:
            return pickle.load(file)
    else:
        return None


def pickle_file_path(folder, identifier: Identifier):
    return os.path.join(folder, f'{identifier.to_filename()}.pickle')


def save_npy(folder, identifier: Identifier, numpy_array):
    """
    Save given image (numpy array) with numpy save function

    :param folder: folder in which to save the given data
    :param identifier: identifier that will be used to create the saved file's filename
    :param numpy_array: numpy array to be saved
    :return: None
    """
    np.save(npy_file_path(folder, identifier), numpy_array)


def load_npy(folder, identifier: Identifier):
    """
    Loads previously saved numpy array

    :param folder: folder in which the data was saved
    :param identifier: data identifier with which it was saved, used to reconstruct the filename
    :return: retrieved numpy array or None if it could not be found
    """
    filepath = npy_file_path(folder, identifier)
    if os.path.exists(filepath):
        return np.load(filepath)
    else:
        return None


def npy_file_path(folder, identifier: Identifier):
    return os.path.join(folder, f'{identifier.to_filename()}.npy')


# Preprocessing sub-steps


def create_labels(
        identifiers: List[EyeIdentifier],
        output_folder,
        ellipse='outer'
):
    """
    Creates binary label images (fixation/no fixation) from MAIA images by extracting the fixation ellipse areas.
    Note: in case appropriate files are not found, the labels for the respective identifiers are not created.

    :param identifiers: eye identifiers for which to create labels
    :param output_folder: folder in which the created labels will be saved
    :param ellipse: 'outer' or 'inner', corresponds to two ellipses that can be found in MAIA images
    :return: None
    """
    assert ellipse in {'outer', 'inner'}

    for identifier in tqdm(identifiers):
        try:
            outer_ellipse, inner_ellipse = labels.extract_labels_from_maia(*identifier)
        except FileNotFoundError:
            continue

        if ellipse == 'outer':
            label = outer_ellipse
        else:
            label = inner_ellipse

        save_npy(output_folder, identifier, label)


def compute_coregistration_parameters(
        identifiers: List[PartIdentifier],
        coregistration_data_output_folder,
        homography_transform_output_folder
):
    """
    Estimates homography transformation between Spectralis and MAIA data by registering Spectralis and MAIA fundus
    images. Uses SIFT as the principle backbone of the algorithm (applied approach from the literature).

    :param identifiers: part identifiers for which the registration parameters should be computed
    :param coregistration_data_output_folder: folder in which the coregistration data will be saved
    :param homography_transform_output_folder: folder in which the estimated homography transforms will be saved
    :return: part identifiers for which the coregistration was done (note: it might not have been done for instance if
        the Spectralis fundus image was not found in its original volume folder, even though a different one could be
        available elsewhere in the raw data, there might have been a few such cases)
    """
    coregistered_identifiers = []
    for identifier in tqdm(identifiers):
        coregistration_data = coregister.compute_coregistration_parameters(*identifier)
        if coregistration_data is not None:
            homography_transform = coregister.homography_transform(coregistration_data)
            save_pickle(coregistration_data_output_folder, identifier, coregistration_data)
            save_pickle(homography_transform_output_folder, identifier, homography_transform)
            coregistered_identifiers.append(identifier)

    return coregistered_identifiers


def filter_identifiers_based_on_coregistration_ssim_measure(
        identifiers: List[PartIdentifier],
        coregistration_data_folder,
        ssim_threshold: float = 0.2,
        window_size_fraction: float = 0.1
) -> List[PartIdentifier]:
    """
    Filters part identifiers by imposing a requirement of minimal coregistration quality measured by the SSIM metric
    between the original Spectralis fundus image and warped MAIA fundus image.

    :param identifiers: part identifiers to filter
    :param coregistration_data_folder: folder in which the coregistration data has been stored
    :param ssim_threshold: minimal SSIM score coregistration must have achieved to not be discarded
    :param window_size_fraction: parameter for SSIM score computation: fraction of the image side that corresponds to
        the window size used in SSIM score computation
    :return: part identifiers filtered based on the SSIM score
    """
    def is_with_sufficient_coregistration_quality(id: PartIdentifier):
        coregistration_data = load_pickle(coregistration_data_folder, id)
        oct_fundus = util.standardize(dataset.image(
            dataset.image_path(id.subject_id, id.group, 'spectralis', 'fundus', id.eye, id.part)
        ))
        maia_fundus = util.standardize(dataset.image(
            dataset.image_path(id.subject_id, id.group, 'maia', 'fundus', id.eye, id.part)
        ))

        homography_transform = coregister.homography_transform(coregistration_data)
        window_size = util.make_odd(int(oct_fundus.shape[0] * window_size_fraction))
        ssim = coregister.ssim_coregistration_quality(oct_fundus, maia_fundus, homography_transform, window_size)
        return ssim >= ssim_threshold

    return [identifier for identifier in tqdm(identifiers) if is_with_sufficient_coregistration_quality(identifier)]


def create_warped_labels(
        identifiers: List[PartIdentifier],
        labels_folder,
        coregistration_data_folder,
        homography_transform_folder,
        output_folder
):
    """
    Warps labels extracted from MAIA images using estimated homography transforms

    :param identifiers: part identifiers for which the labels should be warped
    :param labels_folder: folder in which the extracted label images have been stored
    :param coregistration_data_folder: folder in which the coregistration data has been stored
    :param homography_transform_folder: folder in which the estimated homography transforms have been stored
    :param output_folder: folder in which the warped labels will be saved
    :return: part identifiers for which the warped labels contained some area of fixation (subset of the given part
        identifiers)
    """
    result_identifiers = []
    for identifier in tqdm(identifiers):
        label = load_npy(labels_folder, identifier.to_eye_identifier())
        coregistration_data = load_pickle(coregistration_data_folder, identifier)
        homography_transform = load_pickle(homography_transform_folder, identifier)
        target_shape = (coregistration_data['oct_side'],) * 2
        warped_label = coregister.apply_warp_transform(label, homography_transform, target_shape, order=0)

        if np.count_nonzero(warped_label) > 0:
            result_identifiers.append(identifier)
            save_npy(output_folder, identifier, warped_label)

    return result_identifiers


part_to_volume_type = {
    'full': 'dense',
    'up': 'detail',
    'center': 'detail',
    'down': 'detail'
}


def get_volume_identifiers(
        identifiers: List[PartIdentifier]
):
    """
    Aggregates part identifiers into volume identifiers

    :param identifiers: part identifiers to aggregate
    :return: volumes identifiers resulting from combining given part identifiers
    """
    volume_parts_per_identifier = {}
    for identifier in identifiers:
        eye_id = identifier.to_eye_identifier()
        eye_key = (eye_id.group, eye_id.subject_id, eye_id.eye)
        already_assigned = volume_parts_per_identifier.get(eye_key, [])
        if len(already_assigned) == 0:
            volume_parts_per_identifier[eye_key] = [identifier.part]
        elif len(already_assigned) == 1 and already_assigned[0] == 'full':
            volume_parts_per_identifier[eye_key] = [identifier.part]
        elif identifier.part != 'full':
            volume_parts_per_identifier[eye_key].append(identifier.part)

    return [
        VolumesIdentifier(*eye_key, part_to_volume_type[parts[0]], parts)
        for eye_key, parts in volume_parts_per_identifier.items()
    ]


def pair_octs_with_labels(
        identifiers: List[VolumesIdentifier],
        warped_labels_folder,
        homography_transforms_folder,
        output_folder
):
    """
    Pairs OCT B-scans with labels value sequence (fixation/no fixation) obtained from the warped label images. As the
    result each location on the B-scan along the horizontal direction can be matched with a fixation/no fixation
    label. The resulting set of labelled B-scans is limited to the subset of all B-scans in given volumes - only the
    B-scans intersecting the area representing the fixation can be found in the result.

    :param identifiers: volume identifiers for which the B-scans should be combined with labels
    :param warped_labels_folder: folder in which the warped labels have been saved
    :param homography_transforms_folder: folder in which the estimated homography transforms have been saved
    :param output_folder: folder in which the B-scans representation with accompanying label information will be stored
    :return: None
    """
    for identifier in tqdm(identifiers):
        label_images, homography_transforms = {}, {}
        for part_identifier in identifier.part_identifiers():
            label_images[part_identifier.part] = load_npy(warped_labels_folder, part_identifier)
            homography_transforms[part_identifier.part] = load_pickle(homography_transforms_folder, part_identifier)

        labelled_octs = pairing.pair_octs_with_labels(
            *identifier, identifier.parts, label_images, homography_transforms
        )
        try:
            print(labelled_octs[0].image().shape)
            print(output_folder)
        except:
            continue
        save_pickle(output_folder, identifier, labelled_octs)


class BScanData:
    def __init__(
            self,
            image: np.ndarray,
            labelled_segmentation: np.ndarray,
            thickness: Dict[Tuple[str, str], np.ndarray]
    ):
        """
        Base class for representing a B-scan or its portion together with labelled segmentation map and retinal
        thickness values.

        :param image: OCT B-scan or part of it
        :param labelled_segmentation: labelled segmentation map with different labels for different retinal layers
        :param thickness: retinal thickness values for all the considered layers and the total retinal thickness
        """
        self._image = image
        self._segmentation = labelled_segmentation
        self._thickness = thickness

    def image(self) -> np.ndarray:
        return self._image

    def labelled_segmentation(self) -> np.ndarray:
        return self._segmentation

    def segmentation(self, inner_layer: str, outer_layer: str) -> np.ndarray:
        """
        Binary segmentation mask for the retinal layer between the two given retinal layer boundaries. The retinal layer
        boundary name can be one of (from the most inner to the most outer): 'INFL', 'ONFL', 'IPL', 'OPL', 'ICL',
        'RPE'.

        Segmentation can only be given for the area between two neighbouring retinal layer boundaries, e.g. 'ONFL'
        (inner) and 'IPL' (outer).

        :param inner_layer: name of the inner (in the image: upper) retinal layer boundary
        :param outer_layer: name of the outer (in the image: lower) retinal layer boundary
        :return: binary segmentation mask for the area between the given inner and outer retinal layer boundary
        """
        layer_key = (inner_layer, outer_layer)
        if layer_key in segmentation.LAYERS_TO_LABELS:
            return self._segmentation == segmentation.LAYERS_TO_LABELS[layer_key]
        else:
            raise ValueError(f'No segmentation for layer between {inner_layer} and {outer_layer}')

    def thickness(self, inner_layer: str, outer_layer: str) -> np.ndarray:
        """
        Absolute (in pixels) retinal thickness values sequence for the retinal layer between the two given retinal
        layer boundaries. The retinal layer boundary name can be one of (from the most inner to the most outer):
        'INFL', 'ONFL', 'IPL', 'OPL', 'ICL', 'RPE'.

        Retinal thickness can only be given for the layer between two neighbouring retinal layer boundaries, e.g. 'ONFL'
        (inner) and 'IPL' (outer).

        :param inner_layer: name of the inner (in the image: upper) retinal layer boundary
        :param outer_layer: name of the outer (in the image: lower) retinal layer boundary
        :return: one-dimensional array of absolute (in pixels) retinal thickness values
        """
        if inner_layer is None and outer_layer is None:
            return self._thickness[('INFL', 'RPE')]  # return total thickness
        elif inner_layer is not None and outer_layer is not None and (inner_layer, outer_layer) in self._thickness:
            return self._thickness[(inner_layer, outer_layer)]
        else:
            raise ValueError(f'No thickness for layer between {inner_layer} and {outer_layer}')


class StripeSample(BScanData):
    def __init__(
            self,
            image: np.ndarray,
            labelled_segmentation: np.ndarray,
            thickness: Dict[Tuple[str, str], np.ndarray],
            label: int
    ):
        super().__init__(image, labelled_segmentation, thickness)
        self._label = label

    def label(self):
        return self._label


class PreprocessedOCTScan(BScanData):
    def __init__(
            self,
            identifier: EyeIdentifier,
            scan: dataset.LabelledOCTScan,
            flattened_oct_image: np.ndarray,
            flattened_labelled_segmentation: np.ndarray,
            thickness: Dict[Tuple[str, str], np.ndarray]
    ):
        """
        OCT B-scan data combining:

        * horizontally restricted to the area of interest (based on the fixation area) B-scan to which the retinal
          flattening has been applied,
        * corresponding labelled segmentation mask with each segmented retinal layer labeled with a distinct value,
        * retinal thickness values for all retinal layers plus the total retinal thickness

        NOTE: all the functions providing access to elements of this preprocessed OCT scan return data **restricted
        horizontally to the area concerned**. The extent of this area was obtained from the fixation area and
        horizontally surrounding area of no fixation in the warped label images derived from MAIA.

        :param identifier: eye identifier representing the subject's eye to which this preprocessed B-scan belongs
        :param scan: original labelled B-scan that was used to produce this preprocessed B-scan. Contains e.g. some
            meta information such as original filename
        :param flattened_oct_image: image representing the B-scan after the application of retinal flattening
        :param flattened_labelled_segmentation: labelled segmentation mask corresponding to the flattened OCT image
            with each retinal layer labeled with a distinct value
        :param thickness: a mapping between pairs of neighbouring retinal layer boundaries and the sequence of
            retinal thickness values of the corresponding retinal thickness layer
        """
        super().__init__(flattened_oct_image, flattened_labelled_segmentation, thickness)
        self._identifier = identifier
        self._scan = scan
    
    def scan_data(self):
        return self._scan

    def image(self):
        return super().image()[:, self._scan.left_limit:(self._scan.right_limit + 1)]

    def labelled_segmentation(self):
        return super().labelled_segmentation()[:, self._scan.left_limit:(self._scan.right_limit + 1)]

    def label(self) -> np.ndarray:
        return self._scan.label
    
    def full_label(self) -> np.ndarray:
        return self._scan.full_label
    
    def segmentation(self, inner_layer: str, outer_layer: str):
        return super().segmentation(inner_layer, outer_layer)[:, self._scan.left_limit:(self._scan.right_limit + 1)]

    def thickness(self, inner_layer: str, outer_layer: str):
        return super().thickness(inner_layer, outer_layer)[self._scan.left_limit:(self._scan.right_limit + 1)]
    
    def thicknesses(self):
        thicknesses = np.empty((len(segmentation.RETINAL_LAYERS)-1,self.image().shape[1]))
        for i in range(len(segmentation.RETINAL_LAYERS) - 1):
            thicknesses[i,:] = self.thickness(segmentation.RETINAL_LAYERS[i],segmentation.RETINAL_LAYERS[i+1])
        return thicknesses

    def to_stripe_samples(self, stripe_width_in_degrees_of_visual_angle: float) -> List[StripeSample]:
        """
        Prototype (TODO: should be checked if works correctly):
        Cut the B-scan into stripes of given width to produce samples for input to the CNN.

        :param stripe_width_in_degrees_of_visual_angle: width of the stripe in visual angle degrees. The width in
            pixels is calculated accordingly based on the B-scan origin ('dense' or 'detail' scan) and given width in
            degrees of the visual angle
        :return: a list of stripe samples obtained from this scan
        """
        if self._scan.part != 'full':
            stripe_width = int(dataset.DETAIL_VA.to_length(stripe_width_in_degrees_of_visual_angle))
        else:
            stripe_width = int(dataset.DENSE_VA.to_length(stripe_width_in_degrees_of_visual_angle))
        # This way takes stripes only fully contained in the used OCT part, could be modified to exceed this used OCT
        # part if necessary/desired
        n_stripes = self._scan.width() // stripe_width
        print(n_stripes)
        stripe_samples = []
        for n in range(int(n_stripes)):
            stripe_offset = int(n * stripe_width)
            left_limit = int(self._scan.left_limit + stripe_offset)
            right_limit = int(left_limit + stripe_width)

            stripe_samples.append(StripeSample(
                self._image[:, left_limit:right_limit],
                self._segmentation[:, left_limit:right_limit],
                self._thickness_stripe(left_limit, right_limit),
                self._majority_label(stripe_offset, stripe_width)
            ))
        return stripe_samples

    def _thickness_stripe(self, left_limit, right_limit):
        thickness_stripe = {}
        for layer_key, thickness_values in self._thickness.items():
            thickness_stripe[layer_key] = thickness_values[left_limit:right_limit]
        return thickness_stripe

    def _majority_label(self, stripe_offset, stripe_width):
        labels_array = self._scan.label[stripe_offset:(stripe_offset + stripe_width)]
        if np.count_nonzero(labels_array) > (stripe_width / 2):
            return 1  # fixation stripe
        else:
            return 0  # fundus stripe


def part_identifiers_to_unique_eye_identifiers(identifiers: List[PartIdentifier]):
    """
    Transforms a list of part identifiers into a list of unique corresponding eye identifiers

    :param identifiers: part identifiers to convert
    :return: a list of unique eye identifiers corresponding to given eye identifiers
    """
    id_set = set({})
    results = []
    for id in identifiers:
        eye_key = (id.group, id.subject_id, id.eye)
        if eye_key not in id_set:
            results.append(id.to_eye_identifier())
            id_set.add(eye_key)
    return results


def relative_oct_scan_filepath(full_file_path):
    return os.path.relpath(full_file_path, dataset.DATASET['dataPath'])  # https://stackoverflow.com/a/19856910


def find_meta_file_path(meta_folder, meta_file_names):
    for file_name in meta_file_names:
        path = os.path.join(meta_folder, file_name)
        if os.path.exists(path):
            return path
    return None

def used_oct_scan_part_limits(labelled_oct):
    nonzero_indices = np.nonzero(labelled_oct.label)[0]
    n_fixation_pixels = len(nonzero_indices)
    n_left_fundus_pixels = n_fixation_pixels // 2
    if nonzero_indices[0] - n_left_fundus_pixels < 0:
        n_left_fundus_pixels = nonzero_indices[0]
    n_right_fundus_pixels = n_fixation_pixels - n_left_fundus_pixels
    
    left_limit = nonzero_indices[0] - n_left_fundus_pixels
    right_limit = min(len(labelled_oct.label) - 1, nonzero_indices[-1] + n_right_fundus_pixels)
    labelled_oct.left_limit = left_limit
    labelled_oct.right_limit = right_limit
    
    labelled_oct.label = labelled_oct.label[left_limit: right_limit + 1]
    
    return labelled_oct
    
def create_preprocessed_oct_scans(
        identifiers: List[EyeIdentifier],
        labelled_octs_folder,
        meta_folder,
        filename_to_meta_mapping: Dict[str, str],
        output_folder
):
    """
    Given OCT B-scans paired with labels produces preprocessed OCT B-scans (see PreprocessedOCTScan).

    :param identifiers: eye identifiers for which the preprocessed OCT B-scans should be created
    :param labelled_octs_folder: folder in which the B-scans representation with accompanying label information has
        been stored (the output of pairing OCTs with labels)
    :param meta_folder: folder in which the segmentation meta files split into two sub-folders ('amd' and 'control')
        are located
    :param filename_to_meta_mapping: mapping between relative filepath to the original OCT file and its meta file
        contaiaing the segmentation data
    :param output_folder: folder in which the preprocessed OCT B-scans will be stored
    :return: a tuple with two lists: missing mappings - labelled OCT B-scans data for scans whose meta filename was
        not present in filename_to_meta_mapping; nonexistent meta files - meta filenames that were expected to be
        found based on the filename_to_meta_mapping but where not found under the meta folder hierarchy. TODO:
        eventually there should be no missing mappings nor nonexistent meta files while the missing files are
        found/segmetnated and their data added to the appropriate meta folder and recorded in the mapping
    """
    n_total_octs = 0
    n_missing_oct_segmentations = 0
    missing_mappings = []
    nonexistent_meta_files = []
    cnt = 0
   
    for identifier in tqdm(identifiers):
        preprocessed_oct_scans = []
        labelled_octs = load_pickle(labelled_octs_folder, identifier)
                
        if labelled_octs is None:
            continue
        
        for labelled_oct in labelled_octs:
            
            #MAURO QUI RIBALANCING 
            # labelled_oct = used_oct_scan_part_limits(labelled_oct)
            
            #END REBALANCING
            
#             if cnt < 5:
#                 print(labelled_oct.label.shape)
#                 print(labelled_oct.image().shape)
            n_total_octs += 1
            labelled_oct_path = relative_oct_scan_filepath(labelled_oct.oct_scan.filename)
            
            #Mauro: Fix problem related to mac / windows, maybe a better solution would be to implement it when reading the meta mapping file                
            labelled_oct_path = labelled_oct_path.replace('\\','/')
            #EndMauro
            
            meta_file_names = filename_to_meta_mapping.get(labelled_oct_path, [])
            meta_file_path = find_meta_file_path(meta_folder, meta_file_names)
            
            missing_mapping = len(meta_file_names) == 0
            meta_file_exists = meta_file_path is not None

            if missing_mapping:
                missing_mappings.append((labelled_oct_path, meta_file_names))
            elif not meta_file_exists:
                nonexistent_meta_files.append((labelled_oct_path, meta_file_names))
            if not missing_mapping and meta_file_exists:
                layer_values = segmentation.correct_overlapping_layer_values(
                    segmentation.parse_meta_file(
                        meta_file_path,
                        segmentation.RETINAL_LAYERS
                    ),
                    segmentation.RETINAL_LAYERS
                )
                print(f'MAURO: {labelled_oct.oct_scan.image().shape}')
                # I think I will move the normalization out of the preprocessing for the moment to investigating which gives best results
                #Z_score_Normalization
                #normalized_oct_scan = dataset.z_score(labelled_oct.oct_scan.image())
                #Histogram_Equalization
                #normalized_oct_scan = cv.equalizeHist(labelled_oct.oct_scan.image())
                normalized_oct_scan = labelled_oct.oct_scan.image()

                computed_retinal_flattening = segmentation.compute_retinal_flattening(layer_values)

                flattened_oct_image = segmentation.apply_retinal_flattening(
                    normalized_oct_scan, computed_retinal_flattening
                )
                segmentations = segmentation.compute_retinal_segmentation(
                    normalized_oct_scan, layer_values, segmentation.RETINAL_LAYERS
                )
                labelled_segmentation = segmentation.merge_segmentations_into_labelled_image(
                    segmentations, normalized_oct_scan.shape
                )
                flattened_labelled_segmentation = segmentation.apply_retinal_flattening(
                    labelled_segmentation, computed_retinal_flattening
                )
                thickness = segmentation.compute_retinal_thickness(
                    layer_values, segmentation.RETINAL_LAYERS, total=True
                )
                if cnt < 5:
                    RETINAL_LAYERS = ['INFL', 'ONFL', 'IPL', 'OPL', 'ICL', 'RPE']
                    print("here is the oct: ")
                    img = Image.fromarray(labelled_oct.image()).convert('RGB')
                    display(img)
                    
#                     print("here is the layer_values: ",layer_values['INFL'])
#                     print("here is the layer_values: ",layer_values['ONFL'])
#                     print("here is the layer_values: ",layer_values['IPL'])
#                     print("here is the layer_values: ",layer_values['OPL'])
#                     print("here is the layer_values: ",layer_values['ICL'])
#                     print("here is the layer_values: ",layer_values['RPE'])
                    for i in range(len(RETINAL_LAYERS) -1):
                        prev = RETINAL_LAYERS[i]
                        next_ = RETINAL_LAYERS[i+1]
                        print(f"here is the segmentations: {prev} and {next_}")
                        arr = segmentations[(prev, next_)]
                        arr2 = labelled_oct.image()
                        tot = arr*200 + arr2
                        img = Image.fromarray(75*tot)
                        display(img)
                    
                    print("here is the flattened_oct_image: ",flattened_labelled_segmentation.max())
                    
                    img = Image.fromarray(51*flattened_oct_image).convert('RGB')
                    display(img)
                    print("here is the flattened_labelled_segmentation: ",flattened_labelled_segmentation.max())
                    img = Image.fromarray(51*flattened_labelled_segmentation).convert('RGB')
                    display(img)
                preprocessed_oct_scans.append(PreprocessedOCTScan(
                    identifier, labelled_oct, flattened_oct_image, flattened_labelled_segmentation, thickness
                ))
                cnt +=1
            else:
                n_missing_oct_segmentations += 1

        save_pickle(output_folder, identifier, preprocessed_oct_scans)

    print(f'Missed {n_missing_oct_segmentations} from total {n_total_octs}')
    print(f'{len(missing_mappings)} missing mappings and {len(nonexistent_meta_files)} nonexistent meta files')
    return missing_mappings, nonexistent_meta_files


class StripeIdentifier(Identifier):
    def __init__(self, identifier: EyeIdentifier, index: int):
        super().__init__()
        self._identifier = identifier
        self._index = index

    def to_filename(self):
        return f'{self._identifier.to_filename()}_{self._index}'


def generate_stripe_dataset(
        identifiers: List[EyeIdentifier],
        stripe_width_in_degrees_of_visual_angle: float,
        preprocessed_octs_folder,
        output_folder
):
    """
    Prototype (TODO: should be checked if works correctly)
    Generates stripe samples from the preprocessed OCT B-scans

    :param identifiers: eye identifiers for which the stripe samples should be generated
    :param stripe_width_in_degrees_of_visual_angle: width of the stripe in degrees of the visual angle that will be
        converted accordingly to pixels
    :param preprocessed_octs_folder: folder in which
    :param output_folder: folder in which the preprocessed OCT B-scans have been stored
    :return: None
    """
    os.makedirs(os.path.join(output_folder, 'amd'), exist_ok=False)
    os.makedirs(os.path.join(output_folder, 'control'), exist_ok=False)

    for identifier in identifiers:
        stripe_samples = []
        for preprocessed_oct in load_pickle(preprocessed_octs_folder, identifier):
            stripe_samples += preprocessed_oct.to_stripe_samples(stripe_width_in_degrees_of_visual_angle)

        folder = os.path.join(output_folder, identifier.group)
        for index, stripe_sample in enumerate(stripe_samples):
            save_pickle(folder, StripeIdentifier(identifier, index), stripe_sample)


# Code for running the script from command line


def preprocessing_script_arguments():
    parser = argparse.ArgumentParser(description='OCT dataset preprocessing')
    parser.add_argument('dataset_config', help='.txt config file describing access paths to the dataset')
    parser.add_argument('output_config', help='.txt config file describing output paths for the intermediate and '
                                              'final results of preprocessing')
    parser.add_argument('parameters', help='.txt config file with values for preprocessing parameters')
    return parser.parse_args()


def parse_config_file(config_file):
    config_dict = {}
    with open(config_file, 'r') as file:
        for line in file:
            split_line = line.split('=')
            config_dict[split_line[0]] = split_line[1].rstrip('\n')
    return config_dict


def load_filename_to_meta_mapping(file_path):
    with open(file_path, 'rb') as file:
        return pickle.load(file)


if __name__ == '__main__':
    args = preprocessing_script_arguments()

    print('Initializing dataset')
    dataset.initialize(args.dataset_config)

    print('Parsing preprocessing config')
    output_folders = parse_config_file(args.output_config)
    parameters = parse_config_file(args.parameters)
    filename_to_meta_mapping = load_filename_to_meta_mapping(parameters['mapping'])
    eye_identifiers = all_eye_identifiers()
    part_identifiers = all_part_identifiers()

    print('Extracting labels from BCEA images')
    create_labels(eye_identifiers, output_folders['labels'], ellipse=parameters['ellipse'])

    print('Registering Spectralis and MAIA fundus images')
    part_identifiers = compute_coregistration_parameters(
        part_identifiers, output_folders['coregistration'], output_folders['homography']
    )

    print('Filtering data based on registration SSIM quality')
    part_identifiers = filter_identifiers_based_on_coregistration_ssim_measure(
        part_identifiers, output_folders['coregistration'],
        float(parameters['ssim_threshold']), float(parameters['ssim_window_size_fraction'])
    )

    print('Warping labels')
    part_identifiers = create_warped_labels(
        part_identifiers, output_folders['labels'], output_folders['coregistration'],
        output_folders['homography'], output_folders['warped_labels']
    )

    print('Combining OCT scans with labels')
    volume_identifiers = get_volume_identifiers(part_identifiers)
    pair_octs_with_labels(
        volume_identifiers, output_folders['warped_labels'],
        output_folders['homography'], output_folders['labelled_octs']
    )

    print('Preprocessing OCT scans - flattening, segmentation, retinal thickness computation')
    eye_identifiers = part_identifiers_to_unique_eye_identifiers(part_identifiers)
    missing_mappings, nonexistent_meta_files = create_preprocessed_oct_scans(
        eye_identifiers, output_folders['labelled_octs'], parameters['meta'], filename_to_meta_mapping,
        output_folders['preprocessed_octs']
    )
    
    print('Saving information about missing segmentation files')
    with open('missing_mappings_after_restoration.pickle', 'wb') as missing_mappings_file:
        pickle.dump(missing_mappings, missing_mappings_file)
    with open('nonexistent_meta_files_after_restoration.pickle', 'wb') as nonexistent_file:
        pickle.dump(nonexistent_meta_files, nonexistent_file)

    # print('Generating stripe dataset for the CNN model')
    # generate_stripe_dataset(
    #     eye_identifiers, float(parameters['stripe_width_angle_in_degrees']),
    #     output_folders['preprocessed_octs'], output_folders['stripe_dataset']
    # )

