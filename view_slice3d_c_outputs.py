#!/usr/bin/env python3
import argparse
import os
import subprocess
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib.pyplot as plt
import numpy as np
import scipy.io as spio


EXPECTED_RA_SHAPE = (128, 128)
EXPECTED_RV_MAP_SHAPE = (128, 128)
EXPECTED_RV_SHAPE = (128, 128, 2)
DEFAULT_PREFIX = Path("/tmp/slice3d_c_output")


def compile_slice3d_c(repo_dir, binary):
    subprocess.run(
        [
            "gcc",
            "-std=c11",
            "-O2",
            "-Wall",
            "-Wextra",
            str(repo_dir / "slice3d.c"),
            "-lm",
            "-o",
            str(binary),
        ],
        check=True,
    )


def run_c_from_mat(repo_dir, mat_path, prefix):
    adc_data = spio.loadmat(mat_path, squeeze_me=True)["adc_data"]
    adc_data = np.asarray(adc_data, dtype=np.complex64)

    input_path = Path(f"{prefix}_adc_complex64.bin")
    adc_data.tofile(input_path)

    binary = Path(f"{prefix}_slice3d_c")
    compile_slice3d_c(repo_dir, binary)
    subprocess.run([str(binary), str(input_path), str(prefix)], check=True)


def load_outputs(prefix):
    ra_map = np.load(f"{prefix}_ra_map.npy")
    rv_map_path = Path(f"{prefix}_rv_map.npy")
    rv_map = np.load(rv_map_path) if rv_map_path.exists() else None
    rv_slice = np.load(f"{prefix}_rv_slice.npy")
    ra_cfar = np.load(f"{prefix}_ra_cfar.npy")
    rv_cfar = np.load(f"{prefix}_rv_cfar.npy")
    ra_clusters = np.load(f"{prefix}_ra_clusters.npy")
    rv_clusters = np.load(f"{prefix}_rv_clusters.npy")

    if ra_map.shape != EXPECTED_RA_SHAPE:
        raise ValueError(f"Unexpected RA shape {ra_map.shape}, expected {EXPECTED_RA_SHAPE}")
    if rv_map is not None and rv_map.shape != EXPECTED_RV_MAP_SHAPE:
        raise ValueError(f"Unexpected RV map shape {rv_map.shape}, expected {EXPECTED_RV_MAP_SHAPE}")
    if rv_slice.shape != EXPECTED_RV_SHAPE:
        raise ValueError(f"Unexpected RV shape {rv_slice.shape}, expected {EXPECTED_RV_SHAPE}")
    if ra_cfar.shape != EXPECTED_RA_SHAPE:
        raise ValueError(f"Unexpected RA CFAR shape {ra_cfar.shape}, expected {EXPECTED_RA_SHAPE}")
    if rv_cfar.shape != EXPECTED_RV_MAP_SHAPE:
        raise ValueError(f"Unexpected RV CFAR shape {rv_cfar.shape}, expected {EXPECTED_RV_MAP_SHAPE}")
    if ra_clusters.shape != EXPECTED_RA_SHAPE:
        raise ValueError(f"Unexpected RA clusters shape {ra_clusters.shape}, expected {EXPECTED_RA_SHAPE}")
    if rv_clusters.shape != EXPECTED_RV_MAP_SHAPE:
        raise ValueError(f"Unexpected RV clusters shape {rv_clusters.shape}, expected {EXPECTED_RV_MAP_SHAPE}")

    if rv_map is None:
        rv_map = rv_slice[:, :, 0]

    return (
        ra_map,
        rv_map,
        rv_slice,
        ra_cfar.astype(bool),
        rv_cfar.astype(bool),
        ra_clusters.astype(bool),
        rv_clusters.astype(bool),
    )


def plot_detection_overlay(ax, data, cfar, clusters, title, origin="lower"):
    ax.imshow(data, origin=origin)
    cfar_overlay = np.ma.masked_where(~cfar, cfar)
    ax.imshow(cfar_overlay, origin=origin, cmap="Reds", alpha=0.45)

    centers = np.argwhere(clusters)
    if centers.size:
        ax.scatter(
            centers[:, 1],
            centers[:, 0],
            s=80,
            marker="o",
            facecolors="none",
            edgecolors="cyan",
            linewidths=1.8,
        )

    ax.set_title(f"{title} Overlay ({int(np.count_nonzero(clusters))})")


def plot_outputs(
    ra_map,
    rv_map,
    rv_slice,
    ra_cfar,
    rv_cfar,
    ra_clusters,
    rv_clusters,
    title,
    save_path=None,
    show_rv1=False,
):
    n_cols = 3 if show_rv1 else 2
    fig, axes = plt.subplots(2, n_cols, tight_layout=True)
    fig.suptitle(title)

    axes[0, 0].imshow(ra_map, origin="lower")
    axes[0, 0].set_title("RA Slice")

    axes[0, 1].imshow(rv_map, origin="lower")
    axes[0, 1].set_title("RV Slice")

    plot_detection_overlay(axes[1, 0], ra_map, ra_cfar, ra_clusters, "RA", origin="lower")
    plot_detection_overlay(axes[1, 1], rv_map, rv_cfar, rv_clusters, "RV", origin="lower")

    if show_rv1:
        axes[0, 2].imshow(rv_slice[:, :, 1], origin="lower")
        axes[0, 2].set_title("RV Slice 1")
        axes[1, 2].axis("off")

    if save_path is not None:
        fig.savefig(save_path, dpi=150)
        print(f"Saved image: {save_path}")
    else:
        plt.show()


def compare_with_python_golden(mat_path, ra_map, rv_map, ra_cfar, rv_cfar, ra_clusters, rv_clusters):
    import slice3d

    adc_data = spio.loadmat(mat_path, squeeze_me=True)["adc_data"]
    adc_data = np.asarray(adc_data, dtype=np.complex64)
    range_data = slice3d.range_fft(adc_data)

    rv_slice_py = slice3d.produce_RV_slice(range_data)
    rv_map_py = rv_slice_py[:, :, 0].astype(np.float32)
    ra_slice_py = slice3d.produce_RA_slice(range_data)
    ra_map_py = np.sqrt(ra_slice_py[:, :, 0, 0] ** 2 + ra_slice_py[:, :, 0, 1] ** 2).astype(np.float32)

    ra_cfar_py = slice3d.cfar2(
        ra_map_py,
        guard_cells_row=slice3d.cfar_guard_cells_row,
        guard_cells_col=slice3d.cfar_guard_cells_col,
        training_cells_row=slice3d.cfar_training_cells_row,
        training_cells_col=slice3d.cfar_training_cells_col,
        threshold_scale_row=slice3d.cfar_threshold_scale_row,
        threshold_scale_col=slice3d.cfar_threshold_scale_col,
    )
    rv_cfar_py = slice3d.cfar2(
        rv_map_py,
        guard_cells_row=slice3d.cfar_guard_cells_row,
        guard_cells_col=slice3d.cfar_guard_cells_col,
        training_cells_row=slice3d.cfar_training_cells_row,
        training_cells_col=slice3d.cfar_training_cells_col,
        threshold_scale_row=slice3d.cfar_threshold_scale_row,
        threshold_scale_col=slice3d.cfar_threshold_scale_col,
    )
    ra_clusters_py, _ = slice3d.cluster_detections(
        ra_cfar_py,
        ra_map_py,
        slice3d.cluster_radius,
        slice3d.cluster_min_size,
    )
    rv_clusters_py, _ = slice3d.cluster_detections(
        rv_cfar_py,
        rv_map_py,
        slice3d.cluster_radius,
        slice3d.cluster_min_size,
    )

    for name, ref, got in (("RA map", ra_map_py, ra_map), ("RV map", rv_map_py, rv_map)):
        abs_err = np.abs(ref - got)
        rel_err = abs_err / np.maximum(np.abs(ref), 1e-6)
        print(f"{name} max_abs={abs_err.max():.6g} mean_abs={abs_err.mean():.6g} max_rel={rel_err.max():.6g}")

    for name, ref, got in (("RA CFAR", ra_cfar_py, ra_cfar), ("RV CFAR", rv_cfar_py, rv_cfar)):
        mismatches = np.count_nonzero(ref != got)
        print(
            f"{name} mismatches={mismatches} "
            f"python_hits={np.count_nonzero(ref)} c_hits={np.count_nonzero(got)}"
        )

    for name, ref, got in (("RA clusters", ra_clusters_py, ra_clusters), ("RV clusters", rv_clusters_py, rv_clusters)):
        mismatches = np.count_nonzero(ref != got)
        print(
            f"{name} mismatches={mismatches} "
            f"python_clusters={np.count_nonzero(ref)} c_clusters={np.count_nonzero(got)}"
        )


def main():
    parser = argparse.ArgumentParser(description="Compile/run slice3d.c and visualize RA/RV outputs")
    parser.add_argument(
        "--mat",
        default=None,
        help="Input .mat file. Defaults to the first .mat in slice_sample_data",
    )
    parser.add_argument(
        "--prefix",
        default=str(DEFAULT_PREFIX),
        help="Output prefix for generated .bin/.npy files",
    )
    parser.add_argument(
        "--save",
        default=None,
        help="Optional image path. If omitted, opens an interactive window",
    )
    parser.add_argument(
        "--no-run-c",
        action="store_true",
        help="Only load existing <prefix>_ra_map.npy and <prefix>_rv_slice.npy",
    )
    parser.add_argument(
        "--show-rv1",
        action="store_true",
        help="Also plot the second RV window. slice3d.py plots only RV slice 0",
    )
    parser.add_argument(
        "--no-compare",
        action="store_true",
        help="Skip Python golden comparison",
    )
    args = parser.parse_args()

    repo_dir = Path(__file__).resolve().parent
    prefix = Path(args.prefix)

    if not args.no_run_c:
        if args.mat is None:
            mat_files = sorted((repo_dir / "slice_sample_data").glob("*.mat"))
            if not mat_files:
                raise FileNotFoundError("No .mat files found in slice_sample_data")
            mat_path = mat_files[0]
        else:
            mat_path = Path(args.mat)

        print(f"Compiling and running slice3d.c with: {mat_path}")
        run_c_from_mat(repo_dir, mat_path, prefix)
    elif args.mat is not None:
        mat_path = Path(args.mat)
    else:
        mat_path = None

    ra_map, rv_map, rv_slice, ra_cfar, rv_cfar, ra_clusters, rv_clusters = load_outputs(prefix)
    print(f"RA map: {ra_map.shape} {ra_map.dtype}")
    print(f"RV map: {rv_map.shape} {rv_map.dtype}")
    print(f"RV slice: {rv_slice.shape} {rv_slice.dtype}")
    print(f"RA CFAR: {ra_cfar.shape} hits={np.count_nonzero(ra_cfar)}")
    print(f"RV CFAR: {rv_cfar.shape} hits={np.count_nonzero(rv_cfar)}")
    print(f"RA clusters: {ra_clusters.shape} count={np.count_nonzero(ra_clusters)}")
    print(f"RV clusters: {rv_clusters.shape} count={np.count_nonzero(rv_clusters)}")

    if not args.no_compare and mat_path is not None:
        compare_with_python_golden(mat_path, ra_map, rv_map, ra_cfar, rv_cfar, ra_clusters, rv_clusters)

    plot_outputs(
        ra_map,
        rv_map,
        rv_slice,
        ra_cfar,
        rv_cfar,
        ra_clusters,
        rv_clusters,
        f"C outputs: {prefix}",
        args.save,
        args.show_rv1,
    )


if __name__ == "__main__":
    main()
