import cv2
import dataset
import registration
import os
import pickle
import util

from skimage.exposure import exposure
from skimage.transform import ProjectiveTransform, warp
from skimage.metrics import structural_similarity as ssim
from tqdm import tqdm


# OUTPUT_PATH = dataset.DATASET['coregistrationPath']


def coregistration_data_path(group, subject, eye, part):
    return os.path.join(OUTPUT_PATH, group, f'{subject}_{eye}_{part}.pickle')


def used_data_identifiers():
    def used_coregistration_data(identifier):
        group, _, subject, eye, part = identifier
        file_exists = os.path.exists(coregistration_data_path(group, subject, eye, part))

        if file_exists:
            if part == 'full':
                # no detail volume exists
                return not any([os.path.exists(coregistration_data_path(group, subject, eye, detail_part))
                                for detail_part in dataset.PARTS_DETAIL])
            else:
                return True
        else:
            return False

    return list(filter(used_coregistration_data, dataset.all_data_identifiers()))


def coregistration_ssim_scores(coregistration_data, window_size_fraction=0.1):
    ssim_scores = {}
    for identifier in used_data_identifiers():
        group, subject_id, subject, eye, part = identifier
        oct_fundus = util.standardize(
            dataset.image(dataset.image_path(subject_id, group, 'spectralis', 'fundus', eye, part))
        )
        maia_fundus = dataset.image(dataset.image_path(subject_id, group, 'maia', 'fundus', eye, part))
        window_size = util.make_odd(int(oct_fundus.shape[0] * window_size_fraction))

        oct_fundus_eq = exposure.equalize_hist(oct_fundus)
        maia_fundus_eq = exposure.equalize_hist(maia_fundus)

        transform = homography_transform(coregistration_data)
        warped_maia = apply_warp_transform(maia_fundus_eq, transform, oct_fundus_eq.shape)
        ssim_scores[identifier] = ssim(warped_maia, oct_fundus_eq, win_size=window_size, data_range=1.0)

    return ssim_scores


def ssim_coregistration_quality(oct_fundus, maia_fundus, transform, window_size, data_range=1.0):
    oct_fundus_eq = exposure.equalize_hist(oct_fundus)
    maia_fundus_eq = exposure.equalize_hist(maia_fundus)
    warped_maia = apply_warp_transform(maia_fundus_eq, transform, oct_fundus_eq.shape)
    return ssim(warped_maia, oct_fundus_eq, win_size=window_size, data_range=data_range)


def identifiers_with_acceptable_coregistration_quality(ssim_scores, ssim_threshold=0.2):
    return [identifier for identifier, ssim_score in ssim_scores.items() if ssim_score >= ssim_threshold]


def save_results(data_dictionary):
    group = data_dictionary['group']
    subject = data_dictionary['subject']
    eye = data_dictionary['eye']
    part = data_dictionary['part']

    filename = coregistration_data_path(group, subject, eye, part)
    with open(filename, 'wb') as file:
        pickle.dump(data_dictionary, file)


def log_statistics(log_file, data_dictionary):
    subject = data_dictionary['subject']
    eye = data_dictionary['eye']
    part = data_dictionary['part']

    n_oct_keypoints = len(data_dictionary['oct_descriptors'])
    n_maia_keypoints = len(data_dictionary['maia_descriptors'])
    n_used_matches = len(data_dictionary['source_points'])
    successful_transformation_estimation = (data_dictionary['transformation'] is not None)

    log_file.write(f'{subject} {eye:5} {part:6} OCT_kpts={n_oct_keypoints:4} MAIA_kpts={n_maia_keypoints:4} '
                   f'used_matches={n_used_matches:3} {"ok" if successful_transformation_estimation else "failed"}\n')


def compute_coregistration_parameters(group, subject_id, eye, part):
    # Read original fundus images
    try:
        oct_fundus = dataset.image(dataset.image_path(subject_id, group=group, eye=eye, part=part))
        oct_side = oct_fundus.shape[0]
        maia_fundus = dataset.image(dataset.image_path(subject_id, group=group, device='maia', eye=eye))
    except FileNotFoundError:
        return None

    # Find keypoints and match descriptors
    oct_keypoints, oct_descriptors = registration.extract_keypoints_and_descriptors(oct_fundus)
    maia_keypoints, maia_descriptors = registration.extract_keypoints_and_descriptors(maia_fundus)
    matches = registration.match_descriptors(oct_descriptors, maia_descriptors)

    # Filter matches
    source_points, destination_points = registration.filter_used_points(oct_keypoints, maia_keypoints, matches)

    return {
        'group': group,
        'subject_id': subject_id,
        'eye': eye,
        'part': part,

        'oct_descriptors': oct_descriptors,
        'maia_descriptors': maia_descriptors,

        'source_points': source_points,
        'destination_points': destination_points,
        'oct_side': oct_side
    }


def read_coregistration_data(group, subject, eye, part):
    path = os.path.join(dataset.DATASET['coregistrationPath'], group, f'{subject}_{eye}_{part}.pickle')
    with open(path, 'rb') as coregistration_file:
        return pickle.load(coregistration_file)


def read_warping_transform(group, subject, eye, part):
    return read_coregistration_data(group, subject, eye, part)['transformation']


def homography_transform(coregistration_data):
    source_points = coregistration_data['source_points']
    destination_points = coregistration_data['destination_points']
    homography_matrix, _ = cv2.findHomography(source_points, destination_points, cv2.RANSAC)
    return ProjectiveTransform(homography_matrix)


def apply_warp_transform(image, transform, target_shape, order=1):
    return warp(image, transform, output_shape=target_shape, order=order, preserve_range=True)


def run():
    successful_coregistrations = 0
    total_coregistrations = 0
    with open(os.path.join(OUTPUT_PATH, 'coregistration.log'), 'w') as log_file:
        for group in dataset.GROUPS:
            print(group, 'group:')
            for subject_id, subject in tqdm(enumerate(dataset.subjects(group))):
                for eye in dataset.EYES:
                    for part in dataset.PARTS:
                        data = compute_coregistration_parameters(group, subject, subject_id, eye, part)
                        if data is not None:
                            save_results(data)
                            log_statistics(log_file, data)

                            transformation = registration.create_warping_transformation(
                                data['oct_side'], data['source_points'], data['destination_points']
                            )
                            if transformation is not None:
                                successful_coregistrations += 1
                            total_coregistrations += 1

    print(f'Computed {successful_coregistrations}/{total_coregistrations} coregistrations successfully')


if __name__ == "__main__":
    run()
