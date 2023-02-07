import numpy as np


def parse_layer_values(line):
    return [int(float(value)) for value in line.strip().split(' ')[1:]]


def parse_lines_with_identifiers(lines, identifiers):
    results = {}
    for pattern in identifiers:
        for line in lines:
            if line.startswith(pattern):
                results[pattern] = parse_layer_values(line)

    return results


def get_layer_values(layer_values_dict, layer_name):
    manual_data = f'Default{layer_name}manData'
    if manual_data in layer_values_dict:
        return layer_values_dict[manual_data]
    else:
        return layer_values_dict[f'Default{layer_name}autoData']


def parse_meta_file(meta_filepath, layer_names):
    with open(meta_filepath, 'r') as meta_file:
        meta_lines = meta_file.readlines()

    line_identifiers = []
    for layer in layer_names:
        line_identifiers += [f'Default{layer}manData', f'Default{layer}autoData']
    layer_values_dict = parse_lines_with_identifiers(meta_lines, line_identifiers)

    layer_values = {}
    for layer in layer_names:
        layer_values[layer] = np.array(get_layer_values(layer_values_dict, layer))

    return layer_values


def compute_retinal_flattening(layer_values):
    assert 'RPE' in layer_values, 'layer_values must include values for the RPE layer'
    assert 'INFL' in layer_values, 'layer_values must include values for the INFL layer'

    rpe_values = layer_values['RPE']
    infl_values = layer_values['INFL']

    columns = []
    max_height = -1
    for i, (infl_edge, rpe_edge) in enumerate(zip(infl_values, rpe_values)):
        height = rpe_edge - infl_edge + 1
        column_properties = [infl_edge, rpe_edge, height]
        max_height = max(max_height, height)
        columns.append(column_properties)
    columns = np.array(columns)
    padding = np.vstack((max_height - columns[:, 2], np.zeros((len(columns),), dtype=np.int))).T
    per_column_vertical_limits = columns[:, :2] - padding

    slices = [slice(limit[0], limit[1] + 1) for limit in per_column_vertical_limits]
    return max_height, slices


def apply_retinal_flattening(oct_scan, computed_retinal_flattening):
    out_height, vertical_slices = computed_retinal_flattening
    flattened_oct = np.zeros((out_height, oct_scan.shape[1]))

    for i, vertical_slice in enumerate(vertical_slices):
        flattened_oct[:, i] = oct_scan[vertical_slice, i]
    return flattened_oct


def flatten(oct_scan, layer_values):
    return apply_retinal_flattening(oct_scan, per_column_vertical_limits)


def compute_retinal_segmentation(oct_scan, layer_values, layer_names):
    neighbouring_layer_pairs = zip(layer_names[:-1], layer_names[1:])
    layer_segmentations = {}

    for inner_layer, outer_layer in neighbouring_layer_pairs:
        inner_layer_values = layer_values[inner_layer]
        outer_layer_values = layer_values[outer_layer]

        segmentation = np.zeros_like(oct_scan, dtype=bool)
        for column, (top, bottom) in enumerate(zip(inner_layer_values, outer_layer_values)):
            segmentation[top:bottom, column] = True

        layer_segmentations[(inner_layer, outer_layer)] = segmentation
    return layer_segmentations


# TODO: 1) check if it works
def correct_overlapping_layer_values(layer_values, layer_names):
    conflicting_layer_pairs = zip(layer_names[:-1], layer_names[1:])
    corrected_layer_values = layer_values.copy()
    for inner_layer, outer_layer in conflicting_layer_pairs:
        inner_layer_values = corrected_layer_values[inner_layer]
        outer_layer_values = corrected_layer_values[outer_layer]

        # correct the more outer layer by aligning it with the inner one
        where_overlap = outer_layer_values < inner_layer_values
        corrected_layer_values[outer_layer][where_overlap] = \
            corrected_layer_values[inner_layer][where_overlap]

    return corrected_layer_values


RETINAL_LAYERS = ['INFL', 'ONFL', 'IPL', 'OPL', 'ICL', 'RPE']
LAYERS_TO_LABELS = {
    ('INFL', 'ONFL'): 1,
    ('ONFL', 'IPL'): 2,
    ('IPL', 'OPL'): 3,
    ('OPL', 'ICL'): 4,
    ('ICL', 'RPE'): 5
}


def merge_segmentations_into_labelled_image(layer_segmentations, shape):
    merged = np.zeros(shape, dtype=np.int)
    for layer_key, mask in layer_segmentations.items():
        if np.any(merged[mask]) != 0:
            print('Overriding already assigned layer!')
        merged[mask] = LAYERS_TO_LABELS[layer_key]
    return merged


def compute_retinal_thickness(layer_values, layers, total=True):
    neighbouring_layer_pairs = zip(layers[:-1], layers[1:])
    thickness = {}
    for inner_layer, outer_layer in neighbouring_layer_pairs:
        inner_layer_values = layer_values[inner_layer]
        outer_layer_values = layer_values[outer_layer]

        difference = outer_layer_values - inner_layer_values
        thickness[(inner_layer, outer_layer)] = np.maximum(np.zeros_like(difference), difference)

    if total:
        infl_layer_values = layer_values['INFL']
        rpe_layer_values = layer_values['RPE']
        thickness[('INFL', 'RPE')] = rpe_layer_values - infl_layer_values

    return thickness
