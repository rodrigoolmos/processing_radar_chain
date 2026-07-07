import os
import math
import numpy as np
import scipy.io as spio
import scipy.constants
import scipy.ndimage as ndimage
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

radar_configs = {
    'ramap_rsize': 128,             # RAMap range size
    'ramap_asize': 128,             # RAMap angle size
    'ramap_vsize': 128,             # RAMap velocity size
    'frame_rate': 30,
    'crop_num': 3,                  # crop some indices in range domain
    'n_chirps': 255,                # number of chirps in one frame
    'sample_freq': 4e6,
    'sweep_slope': 21.0017e12,
    'data_type': 'RISEP',           # 'RI': real + imaginary, 'AP': amplitude + phase
    'ramap_rsize_label': 122,       # TODO: to be updated
    'ramap_asize_label': 121,       # TODO: to be updated
    'ra_min_label': -60,            # min radar angle
    'ra_max_label': 60,             # max radar angle
    'rr_min': 1.0,                  # min radar range (fixed)
    'rr_max': 25.0,                 # max radar range (fixed)
    'ra_min': -90,                  # min radar angle (fixed)
    'ra_max': 90,                   # max radar angle (fixed)
    'ramap_folder': 'WIN_HEATMAP',
}

def confmap2ra(radar_configs, name, radordeg=None):
    """
    Map confidence map to range(m) and angle(deg): not uniformed angle
    :param name: 'range' for range mapping, 'angle' for angle mapping
    :return: mapping grids
    """
    # TODO: add more args for different network settings
    Fs = radar_configs['sample_freq']
    sweepSlope = radar_configs['sweep_slope']
    num_crop = radar_configs['crop_num']
    fft_Rang = radar_configs['ramap_rsize'] + 2*num_crop
    fft_Ang = radar_configs['ramap_asize']
    c = scipy.constants.speed_of_light

    if name == 'range':
        freq_res = Fs / fft_Rang
        freq_grid = np.arange(fft_Rang) * freq_res
        rng_grid = freq_grid * c / sweepSlope / 2
        rng_grid = rng_grid[num_crop:fft_Rang - num_crop]
        return rng_grid

    if name == 'angle':
        # for [-90, 90], w will be [-1, 1]
        w = np.linspace(math.sin(math.radians(radar_configs['ra_min'])),
                        math.sin(math.radians(radar_configs['ra_max'])),
                        radar_configs['ramap_asize'])
        if radordeg is None or radordeg == 'deg':
            agl_grid = np.degrees(np.arcsin(w))  # rad to deg
        elif radordeg == 'rad':
            agl_grid = np.arcsin(w)
        else:
            raise TypeError
        return agl_grid

n_angle = 128
n_vel = 128
n_range = 128
n_chirp = 255
n_rx = 8
n_sample = 128
noma_rcs = 30000
range_grid = confmap2ra(radar_configs, name='range')
cfar_guard_cells_row = 20
cfar_guard_cells_col = 2
cfar_training_cells_row = 8
cfar_training_cells_col = 8
cfar_threshold_scale_row = 3
cfar_threshold_scale_col = 3
cluster_radius = 2
cluster_min_size = 1

def produce_RV_slice(data):
    hanning_win = np.hamming(n_vel)
    win_data1 = np.zeros([data.shape[0], data.shape[1], n_vel], dtype=np.complex128)
    win_data2 = np.zeros([data.shape[0], data.shape[1], n_vel], dtype=np.complex128)
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            win_data1[i, j, :] = np.multiply(data[i, j, 0:n_vel], hanning_win)
            win_data2[i, j, :] = np.multiply(data[i, j, n_vel - 1:], hanning_win)

    fft_data_raw1 = np.fft.fft(win_data1, n_vel, axis=2)
    fft_data_raw1 = np.fft.fftshift(fft_data_raw1, axes=2)
    fft3d_data1 = np.sum(np.abs(fft_data_raw1), axis=1) / n_rx
    fft3d_data1 = np.expand_dims(fft3d_data1, axis=2)

    fft_data_raw2 = np.fft.fft(win_data2, n_vel, axis=2)
    fft_data_raw2 = np.fft.fftshift(fft_data_raw2, axes=2)
    fft3d_data2 = np.sum(np.abs(fft_data_raw2), axis=1) / n_rx
    fft3d_data2 = np.expand_dims(fft3d_data2, axis=2)

    # output format [range, velocity, 2chirps]
    fft3d_data = np.float32(np.concatenate((fft3d_data1, fft3d_data2), axis=2))
    return fft3d_data, fft_data_raw1, fft_data_raw2


def produce_VA_slice(rv_raw1, rv_raw2):
    hanning_win = np.hamming(n_rx)
    win_data1 = np.zeros([rv_raw1.shape[0], rv_raw1.shape[1], rv_raw1.shape[2]], dtype=np.complex128)
    win_data2 = np.zeros([rv_raw2.shape[0], rv_raw2.shape[1], rv_raw2.shape[2]], dtype=np.complex128)
    for i in range(rv_raw1.shape[0]):
        for j in range(rv_raw1.shape[2]):
            win_data1[i, :, j] = np.multiply(rv_raw1[i, :, j], hanning_win)
            win_data2[i, :, j] = np.multiply(rv_raw2[i, :, j], hanning_win)

    fft_data_raw1 = np.fft.fft(win_data1, n_angle, axis=1)
    fft3d_data1 = np.sum(np.abs(np.fft.fftshift(fft_data_raw1, axes=1)), axis=0) / rv_raw1.shape[0]
    fft3d_data1 = np.expand_dims(fft3d_data1, axis=2)

    fft_data_raw2 = np.fft.fft(win_data2, n_angle, axis=1)
    fft3d_data2 = np.sum(np.abs(np.fft.fftshift(fft_data_raw2, axes=1)), axis=0) / rv_raw2.shape[0]
    fft3d_data2 = np.expand_dims(fft3d_data2, axis=2)

    # output format [angle, velocity, 2chirps]
    fft3d_data = np.float32(np.concatenate((fft3d_data1, fft3d_data2), axis=2))
    return fft3d_data


def produce_RA_slice(data, filter_static=False, keep_complex=False):
    hanning_win = np.hamming(n_rx)
    win_data = np.zeros([data.shape[0], data.shape[1], data.shape[2]], dtype=np.complex128)
    for i in range(data.shape[0]):
        for j in range(data.shape[2]):
            win_data[i, :, j] = np.multiply(data[i, :, j], hanning_win)

    fft_data_raw = np.fft.fft(win_data, n_angle, axis=1)
    fft3d_data_cmplx = np.fft.fftshift(fft_data_raw, axes=1)
    if keep_complex is True:
        fft3d_data = fft3d_data_cmplx
    else:
        fft_data_real = np.expand_dims(fft3d_data_cmplx.real, axis=3)
        fft_data_imag = np.expand_dims(fft3d_data_cmplx.imag, axis=3)
        # output format [range, angle, chirps, real/imag]
        fft3d_data = np.float32(np.concatenate((fft_data_real, fft_data_imag), axis=3))
    if filter_static:
        fft3d_data = fft3d_data - np.mean(fft3d_data, axis=2, keepdims=True)

    return fft3d_data


def produce_RCSmap(data):
    hanning_win = np.hamming(n_rx)
    win_data = np.zeros([data.shape[0], data.shape[1], data.shape[2]], dtype=np.complex128)
    for i in range(data.shape[0]):
        for j in range(data.shape[2]):
            win_data[i, :, j] = np.multiply(data[i, :, j], hanning_win)

    fft_data_raw = np.fft.fft(win_data, n_angle, axis=1)
    fft3d_data_amp = np.abs(np.fft.fftshift(fft_data_raw, axes=1)) ** 2
    fft3d_data_amp = np.sum(fft3d_data_amp, axis=2) / data.shape[2]
    range_weight = np.tile(np.expand_dims(range_grid, axis=1) ** 4, (1, n_angle))
    rcs_data = np.multiply(range_weight, fft3d_data_amp) / noma_rcs

    return rcs_data


def cfar_1d(data, axis, guard_cells=2, training_cells=8, threshold_scale=4.0):
    data = np.asarray(data)
    detections = np.zeros(data.shape, dtype=bool)
    moved_data = np.moveaxis(data, axis, -1)
    moved_detections = np.moveaxis(detections, axis, -1)
    n_cells = moved_data.shape[-1]

    for idx in np.ndindex(moved_data.shape[:-1]):
        line = moved_data[idx]
        start_cell = guard_cells + training_cells
        end_cell = n_cells - guard_cells - training_cells
        for cell in range(start_cell, end_cell):
            left_start = cell - guard_cells - training_cells
            left_end = cell - guard_cells
            right_start = cell + guard_cells + 1
            right_end = cell + guard_cells + training_cells + 1
            training_data = np.concatenate((line[left_start:left_end], line[right_start:right_end]))

            threshold = np.mean(training_data) * threshold_scale
            moved_detections[idx + (cell,)] = line[cell] > threshold

    return detections


def cfar2(data, guard_cells_row=2, guard_cells_col=2,
           training_cells_row=8, training_cells_col=8,
             threshold_scale_row=4.0, threshold_scale_col=4.0):
    row_detections = cfar_1d(
        data,
        axis=1,
        guard_cells=guard_cells_row,
        training_cells=training_cells_row,
        threshold_scale=threshold_scale_row,
    )
    column_detections = cfar_1d(
        data,
        axis=0,
        guard_cells=guard_cells_col,
        training_cells=training_cells_col,
        threshold_scale=threshold_scale_col,
    )

    return row_detections & column_detections


def cluster_detections(detections, values=None, radius=2, min_size=1):
    detections = np.asarray(detections, dtype=bool)
    clustered = np.zeros(detections.shape, dtype=bool)
    clusters = []

    if not np.any(detections):
        return clustered, clusters

    structure = np.ones((2 * radius + 1, 2 * radius + 1), dtype=bool)
    expanded_detections = ndimage.binary_dilation(detections, structure=structure)
    labels, n_labels = ndimage.label(expanded_detections)

    for label_id in range(1, n_labels + 1):
        coords = np.argwhere(detections & (labels == label_id))
        if coords.shape[0] < min_size:
            continue

        if values is None:
            center = np.rint(np.mean(coords, axis=0)).astype(int)
        else:
            cluster_values = values[coords[:, 0], coords[:, 1]]
            center = coords[np.argmax(cluster_values)]

        center = np.clip(center, [0, 0], np.array(detections.shape) - 1)
        clustered[center[0], center[1]] = True
        clusters.append({
            'center': tuple(int(value) for value in center),
            'size': coords.shape[0],
        })

    return clustered, clusters


def plot_detection_overlay(ax, data, cfar, clusters, title, origin='lower'):
    ax.imshow(data, origin=origin)

    cfar_overlay = np.ma.masked_where(~cfar, cfar)
    ax.imshow(cfar_overlay, origin=origin, cmap='Reds', alpha=0.45)

    if clusters:
        centers = np.array([cluster['center'] for cluster in clusters])
        ax.scatter(
            centers[:, 1],
            centers[:, 0],
            s=80,
            marker='o',
            facecolors='none',
            edgecolors='cyan',
            linewidths=1.8,
        )

    ax.set_title(f"{title} Overlay ({len(clusters)})")


def range_fft(data):
    hanning_win = np.hamming(n_sample)
    win_data = np.zeros([data.shape[0], data.shape[1], data.shape[2]], dtype=np.complex128)
    for i in range(data.shape[1]):
        for j in range(data.shape[2]):
            win_data[:, i, j] = np.multiply(data[:, i, j], hanning_win)
    fft_data_raw = np.fft.fft(win_data, n_range, axis=0)

    return fft_data_raw


def save_ra_slice(data, save_dir_ra, new_file_name):
    for i in range(data.shape[2]):
        save_fod = os.path.join(save_dir_ra, str(i).zfill(4))
        if not os.path.exists(save_fod):
            os.makedirs(save_fod)
        save_dir = os.path.join(save_fod, new_file_name)
        np.save(save_dir, data[:, :, i, :])


def main():
    """
    This function preprocess the raw data and save the data to the local
    Input: raw ADC data
    Output: RA slice (real and imaginary part of the first chirp after the denoise)
    RV slice (accumulate along the Angle domain)
    VA slice (accumulate along the Range domain)
    """
    root_dir = os.path.join(os.path.dirname(__file__), 'slice_sample_data')
    files = sorted(
        os.path.join(root_dir, file_name)
        for file_name in os.listdir(root_dir)
        if file_name.endswith('.mat')
    )

    if not files:
        raise FileNotFoundError(f"No .mat files found in {root_dir}")

    print('Processing ', root_dir)
    for file_dir in files[0:1]:
        file_name = os.path.basename(file_dir)
        mat = spio.loadmat(file_dir, squeeze_me=True)
        data = np.asarray(mat["adc_data"])
        # Range FFT
        data = range_fft(data)
        # generate RV slice
        RV_slice, rv_raw1, rv_raw2 = produce_RV_slice(data) # (128, 128, 2)
        # generate VA slice
        VA_slice = produce_VA_slice(rv_raw1, rv_raw2)   # (128, 128, 2)
        # generate RA slice
        RA_slice = produce_RA_slice(data)   # (128, 128, 255, 2)
        RA_map = np.sqrt(RA_slice[:, :, 0, 0] ** 2 + RA_slice[:, :, 0, 1] ** 2)
        RV_map = RV_slice[:, :, 0]
        VA_map = VA_slice[:, :, 0]
        RA_cfar = cfar2(RA_map, guard_cells_row=cfar_guard_cells_row, guard_cells_col=cfar_guard_cells_col,
                        training_cells_row=cfar_training_cells_row, training_cells_col=cfar_training_cells_col,
                        threshold_scale_row=cfar_threshold_scale_row, threshold_scale_col=cfar_threshold_scale_col)
        RV_cfar = cfar2(RV_map, guard_cells_row=cfar_guard_cells_row, guard_cells_col=cfar_guard_cells_col,
                        training_cells_row=cfar_training_cells_row, training_cells_col=cfar_training_cells_col,
                        threshold_scale_row=cfar_threshold_scale_row, threshold_scale_col=cfar_threshold_scale_col)
        VA_cfar = cfar2(VA_map, guard_cells_row=cfar_guard_cells_row, guard_cells_col=cfar_guard_cells_col,
                        training_cells_row=cfar_training_cells_row, training_cells_col=cfar_training_cells_col,
                        threshold_scale_row=cfar_threshold_scale_row, threshold_scale_col=cfar_threshold_scale_col)
        _, RA_clusters = cluster_detections(RA_cfar, RA_map, cluster_radius, cluster_min_size)
        _, RV_clusters = cluster_detections(RV_cfar, RV_map, cluster_radius, cluster_min_size)
        _, VA_clusters = cluster_detections(VA_cfar, VA_map, cluster_radius, cluster_min_size)

        # Create 2x3 sub plots: maps on top, detections overlaid below.
        gs = gridspec.GridSpec(2, 3)
        fig = plt.figure(tight_layout=True)
        fig.suptitle(f"Processing {file_name}")
        ax = plt.subplot(gs[0, 0])  # row 0, col 0
        plt.imshow(RA_map, origin='lower')
        ax.set_title("RA Slice")

        ax2 = plt.subplot(gs[0, 1])  # row 0, col 1
        plt.imshow(RV_map, origin='lower')
        ax2.set_title("RV Slice")

        ax3 = plt.subplot(gs[0, 2])  # row 0, col 2
        plt.imshow(VA_map)
        ax3.set_title("VA Slice")

        ax = plt.subplot(gs[1, 0])
        plot_detection_overlay(ax, RA_map, RA_cfar, RA_clusters, "RA", origin='lower')

        ax2 = plt.subplot(gs[1, 1])
        plot_detection_overlay(ax2, RV_map, RV_cfar, RV_clusters, "RV", origin='lower')

        ax3 = plt.subplot(gs[1, 2])
        plot_detection_overlay(ax3, VA_map, VA_cfar, VA_clusters, "VA")
        plt.show()


if __name__ == '__main__':
    main()
