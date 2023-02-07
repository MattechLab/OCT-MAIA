import dataset

import numpy as np
import os
import pickle

from skimage import draw
from skimage.morphology import erosion, dilation, selem
from tqdm import tqdm


def try_loading_subject_volume(group, subject_id, eye, part):
    try:
        return dataset.oct_volume(subject_id, group, eye, part)
    except FileNotFoundError:
        return None


def load_subject_volumes(group, subject_id, eye, parts):
    part_volume_tuples = []
    for part in parts:
        try:
            part_volume_tuples.append((part, dataset.oct_volume(subject_id, group, eye, part)))
        except FileNotFoundError:
            continue

    if len(part_volume_tuples) == 0:
        print(f'Failed to find volume for {group} {subject_id} {eye}')
        raise FileNotFoundError

    return part_volume_tuples


def pair_octs_with_labels(group, subject_id, eye, parts, label_images, homography_transforms):
    try:
        part_volume_tuples = load_subject_volumes(group, subject_id, eye, parts)
    except FileNotFoundError:
        return []

    filtered_part_scans_tuples = filter_scans_from_intersecting_volume_regions(
        group, subject_id, eye, part_volume_tuples, homography_transforms
    )

    labelled_octs = []
    for part, oct_scans in filtered_part_scans_tuples:
        
        label_image = label_images[part]
        intersecting_scans = [oct_scan for oct_scan in oct_scans if intersects_bcea(oct_scan, label_image)]

        for oct_scan in intersecting_scans:
            #Still same dimensions
            row, start_col = oct_scan.start

            #left_limit, right_limit = used_oct_scan_part_limits(oct_scan, label_image)
            left_limit = 0
            right_limit = oct_scan.end[1] - oct_scan.start[1]
            #label = label_image[row, (start_col + left_limit):(start_col + right_limit + 1)]
#             print(oct_scan.end[1] - start_col)
            label = label_image[row, start_col : oct_scan.end[1] + 1]
            try:
                print(f'MAURO: {oct_scan.image().shape}')
            except:
                continue
                #Capire cosa non va
            labelled_octs.append(dataset.LabelledOCTScan(
                group, subject_id, eye, part, oct_scan, left_limit, right_limit, label, label_image)
            )
    return labelled_octs


MAIA_SIZE = 1024


# Note: part_volume_tuples must be in up, center, down order
def filter_scans_from_intersecting_volume_regions(group, subject_id, eye, part_volume_tuples, homography_transforms):
    label_space_lines = np.zeros((MAIA_SIZE, MAIA_SIZE), dtype=bool)
    label_space_areas = np.zeros((MAIA_SIZE, MAIA_SIZE), dtype=bool)
    filtered_scans = []
#     cnt = 0
    for part, volume in part_volume_tuples:
        part_filtered_scans = []
        homography_transform = homography_transforms[part]
        for scan in volume.scans:
#             if (cnt % 20 == 0):
#                 print(scan.image().shape)
#             cnt += 1
            rr, cc = draw.line(*scan.start, *scan.end)
            line_coords = np.hstack((rr.reshape(-1, 1), cc.reshape(-1, 1)))
            label_space_line_coords = np.around(homography_transform(line_coords)).astype(int)

            # Filter out elements exceeding the image borders
            label_space_line_coords = label_space_line_coords[
                np.all(label_space_line_coords > 0) & np.all(label_space_line_coords < MAIA_SIZE)
            ]

            # Check if the scan line does not intersect the already covered area
            if not np.any(label_space_areas[label_space_line_coords[:, 0], label_space_line_coords[:, 1]]):
                part_filtered_scans.append(scan)
                label_space_lines[label_space_line_coords[:, 0], label_space_line_coords[:, 1]] = True

        # Adding estimation of the covered area
        corners = np.array([
            list(volume.scans[-1].start), list(volume.scans[-1].end),
            list(volume.scans[0].end), list(volume.scans[0].start)
        ])
        label_space_corners = np.around(homography_transform(corners))
        area_rr, area_cc = draw.polygon(label_space_corners[:, 0], label_space_corners[:, 1])
        exceeding_size_above = max(0, max(area_rr.max(), area_cc.max()) - MAIA_SIZE)
        exceeding_size_below = max(0, -min(area_rr.min(), area_cc.min()))
        pad = np.ceil((exceeding_size_above + exceeding_size_below) / 2).astype(int) * 2

        if pad > 0:
            print(f'Padding {group} subject {subject_id} {eye} eye in {part} part with {pad}')
            padded_label_area = np.pad(label_space_areas, ((pad, pad), (pad, pad)))
            padded_label_area[area_rr + pad, area_cc + pad] = True
            label_space_areas = padded_label_area[pad:-pad, pad:-pad]
            
        else:
            label_space_areas[area_rr, area_cc] = True

        filtered_scans.append((part, part_filtered_scans))
    return filtered_scans


def intersects_bcea(oct_scan, label_image):
    label_slice = label_image[oct_scan.start[0], oct_scan.start[1]:oct_scan.end[1]]
    return np.any(label_slice)


def grow_to_balanced_classes_label(label_image):
    n_fixation = np.count_nonzero(label_image)
    balanced_classes = False
    grown_label_image = np.copy(label_image)

    se_radius = 27
    next_operation = 'dilation'
    prev_abs_difference = 0.5

    while not balanced_classes:
        prev_grown_label_image = np.copy(grown_label_image)
        se = selem.disk(se_radius)
        if next_operation == 'dilation':
            grown_label_image = dilation(grown_label_image, se)
        elif next_operation == 'erosion':
            grown_label_image = erosion(grown_label_image, se)

        n_total_size = np.count_nonzero(grown_label_image)
        n_no_fixation = n_total_size - n_fixation
        difference = 0.5 - n_no_fixation / n_total_size
        abs_difference = abs(difference)

        if difference > 0:
            if next_operation == 'erosion':
                se_radius //= 2
            next_operation = 'dilation'
        else:
            if next_operation == 'dilation':
                se_radius //= 2
            next_operation = 'erosion'

        if se_radius == 0:
            balanced_classes = True
            if prev_abs_difference < abs_difference:
                grown_label_image = prev_grown_label_image

        prev_abs_difference = abs_difference

    return grown_label_image


def used_oct_scan_part_limits(oct_scan, label_image):
#    row = oct_scan.start[0]
    start_col, end_col = 0, oct_scan.end[1] - oct_scan.start[1]
#     #Mauro
#     print('here: ',end_col - start_col)
#     #Mauro
#     label_slice = label_image[row, start_col:(end_col + 1)]
#     nonzero_indices = np.nonzero(label_slice)[0]
#     n_fixation_pixels = len(nonzero_indices)

#     n_left_fundus_pixels = n_fixation_pixels // 2
#     if nonzero_indices[0] - n_left_fundus_pixels < 0:
#         n_left_fundus_pixels = nonzero_indices[0]
#     n_right_fundus_pixels = n_fixation_pixels - n_left_fundus_pixels
    
#     left_limit = nonzero_indices[0] - n_left_fundus_pixels
#     right_limit = min(len(label_slice) - 1, nonzero_indices[-1] + n_right_fundus_pixels)
#     return left_limit, right_limit
    return start_col, end_col


def save_results(group, labelled_octs, filename='labelled_octs.pickle'):
    filename = os.path.join(dataset.DATASET['labelsPath'], group, filename)
    with open(filename, 'wb') as file:
        pickle.dump(labelled_octs, file)


def run(name='labelled_octs'):
    with open(os.path.join(dataset.DATASET['labelsPath'], f'{name}.log'), 'w') as log_file:
        for group in dataset.GROUPS:
            labelled_octs = []
            log_file.write(group + ':\n')

            print(f'{group}:')
            for subject_id, subject in enumerate(tqdm(dataset.subjects(group))):  # https://stackoverflow.com/a/49281428
                for eye in dataset.EYES:
                    try:
                        subject_labelled_octs = pair_octs_with_label_old(group, subject_id, eye)
                        labelled_octs += subject_labelled_octs
                        log_file.write(f'{subject_id} {eye}: {len(subject_labelled_octs)}\n')
                    except FileNotFoundError:
                        log_file.write(f'{subject_id} {eye}: missing\n')

            save_results(group, labelled_octs, filename=f'{name}.pickle')
            log_file.write(f'{group} total: {len(labelled_octs)}\n')


if __name__ == "__main__":
    run(name='labelled_octs_v2')
