"""Generate py_sod_metrics reference values for the birefnet-metric parity tests.

Runs the upstream BiRefNet `evaluation/metrics.py` classes (OpenCV, scikit-image, and tqdm are
replaced by exact NumPy / SciPy equivalents or unused stubs) on two synthetic 8-bit samples and
writes `tests/support/upstream_reference.rs`.

Usage: python gen_upstream_reference.py <BiRefNet>/evaluation/metrics.py
Requires numpy and scipy.
"""

import argparse
import importlib.util
import pathlib
import sys
import types

import numpy as np
from scipy import ndimage

OUT = pathlib.Path(__file__).resolve().parent.parent / "tests" / "support" / "upstream_reference.rs"


def install_shims():
    """Registers stand-ins for the modules `metrics.py` imports but BIoU/F/E/S/WFm do not need."""
    cv2 = types.ModuleType("cv2")
    cv2.BORDER_CONSTANT = 0

    def copy_make_border(img, top, bottom, left, right, border_type, value=0):
        assert border_type == cv2.BORDER_CONSTANT
        return np.pad(img, ((top, bottom), (left, right)), mode="constant", constant_values=value)

    def erode(img, kernel, iterations=1):
        # cv2.erode's default border never lowers values at the image edge (border = +max).
        assert kernel.shape == (3, 3) and kernel.all()
        out = img
        for _ in range(iterations):
            out = ndimage.grey_erosion(out, size=(3, 3), mode="constant", cval=255)
        return out

    cv2.copyMakeBorder = copy_make_border
    cv2.erode = erode
    sys.modules["cv2"] = cv2

    tqdm = types.ModuleType("tqdm")
    tqdm.tqdm = lambda x, *a, **k: x
    sys.modules["tqdm"] = tqdm

    skimage = types.ModuleType("skimage")
    morph = types.ModuleType("skimage.morphology")
    morph.skeletonize = lambda x: x
    morph.disk = lambda r: np.ones((2 * r + 1, 2 * r + 1))
    measure = types.ModuleType("skimage.measure")
    measure.label = lambda *a, **k: None
    skimage.morphology = morph
    skimage.measure = measure
    sys.modules["skimage"] = skimage
    sys.modules["skimage.morphology"] = morph
    sys.modules["skimage.measure"] = measure


def load_upstream(path):
    install_shims()
    spec = importlib.util.spec_from_file_location("upstream_metrics", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def make_samples():
    rng = np.random.default_rng(20261122)
    h, w = 41, 57
    yy, xx = np.mgrid[0:h, 0:w]
    samples = []
    # Soft elliptical blob with noise vs. a slightly shifted binary ellipse.
    blob = np.exp(-(((yy - 18) / 11.0) ** 2 + ((xx - 30) / 16.0) ** 2))
    pred = np.clip(blob * 255 + rng.normal(0, 18, (h, w)), 0, 255).astype(np.uint8)
    gt = ((((yy - 20) / 10.0) ** 2 + ((xx - 28) / 15.0) ** 2) <= 1.0).astype(np.uint8) * 255
    samples.append((pred, gt))
    # Two rectangles with a hole and a ramp prediction touching the image border.
    gt2 = np.zeros((h, w), np.uint8)
    gt2[2:20, 0:25] = 255
    gt2[25:39, 35:57] = 255
    gt2[8:12, 8:14] = 0
    pred2 = np.clip(gt2.astype(np.float64) * 0.8 + xx * 1.5 + rng.uniform(0, 40, (h, w)), 0, 255)
    samples.append((pred2.astype(np.uint8), gt2))
    return h, w, samples


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("metrics_py", help="path to the upstream BiRefNet evaluation/metrics.py")
    m = load_upstream(parser.parse_args().metrics_py)

    h, w, samples = make_samples()
    fm, em, sm, wfm = m.FMeasure(), m.EMeasure(), m.SMeasure(), m.WeightedFMeasure()
    mae, mse, biou = m.MAEMeasure(), m.MSEMeasure(), m.BIoUMeasure()
    per_sample = []
    for pred, gt in samples:
        for metric in (fm, em, sm, wfm, mae, mse, biou):
            metric.step(pred=pred, gt=gt)
        per_sample.append(
            dict(
                mae=mae.maes[-1],
                mse=mse.mses[-1],
                sm=sm.sms[-1],
                wfm=wfm.weighted_fms[-1],
                adp_fm=fm.adaptive_fms[-1],
                adp_em=em.adaptive_ems[-1],
                fm_curve=fm.changeable_fms[-1],
                em_curve=em.changeable_ems[-1],
                biou_curve=biou.bious[-1],
            )
        )

    def arr(name, values, ty="f64"):
        body = ", ".join(repr(float(v)) if ty == "f64" else str(int(v)) for v in values)
        return f"pub const {name}: [{ty}; {len(values)}] = [{body}];\n"

    with open(OUT, "w", newline="\n") as f:
        f.write("//! Reference values generated from the upstream BiRefNet `evaluation/metrics.py`\n")
        f.write("//! (`py_sod_metrics`) by `scripts/gen_upstream_reference.py`. Do not edit by hand.\n\n")
        f.write(f"pub const HEIGHT: usize = {h};\npub const WIDTH: usize = {w};\n\n")
        for i, (pred, gt) in enumerate(samples):
            f.write(arr(f"PRED_{i}", pred.ravel(), "u8"))
            f.write(arr(f"GT_{i}", gt.ravel(), "u8"))
        for i, ref in enumerate(per_sample):
            for key in ("mae", "mse", "sm", "wfm", "adp_fm", "adp_em"):
                f.write(f"pub const {key.upper()}_{i}: f64 = {float(ref[key])!r};\n")
            for key in ("fm_curve", "em_curve", "biou_curve"):
                f.write(arr(f"{key.upper()}_{i}", ref[key]))
            if i + 1 < len(per_sample):
                f.write("\n")
    print("wrote", OUT)


if __name__ == "__main__":
    main()
