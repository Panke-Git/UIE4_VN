"""Numerical and display checks without checkpoints or model construction."""
import numpy as np
from PIL import Image
import pytest

from tools.visualize_v16_anchor_residual import derive, fresh_output, render, signed_display


def test_mixed_precision_keeps_signed_residual_and_original_operands():
    I = np.array([[[[0.1, 0.8]], [[0.6, 0.2]], [[0.5, 0.9]]]], dtype=np.float32)
    b = np.array([[0.3, 0.4, 0.7]], dtype=np.float16)
    R = np.array([[[[0.2, -0.4]], [[-0.5, 0.3]], [[0.6, -0.2]]]], dtype=np.float16)
    originals = [x.copy() for x in (I, b, R)]
    expected_A = I - b[:, :, None, None]
    expected_delta = R * expected_A
    Ic = I + expected_delta
    A, delta, reconstructed, error = derive(I, b, R, Ic)
    np.testing.assert_array_equal(A, expected_A)
    np.testing.assert_array_equal(delta, expected_delta)
    np.testing.assert_array_equal(reconstructed, Ic)
    assert A.dtype == delta.dtype == np.float32 and error == 0
    assert A.min() < 0 and delta.min() < 0
    for x, original in zip((I, b, R), originals):
        np.testing.assert_array_equal(x, original)
    with pytest.raises(ValueError, match="mismatch"):
        derive(I, b, R, Ic + 0.1)


def test_shared_scale_includes_both_arrays_and_all_channels():
    A = np.zeros((1, 3, 2, 4), dtype=np.float32)
    A[0, 0, 0, 0], A[0, 2, 1, 3] = -1, 2
    delta = A * 4
    images, scale = signed_display(A, delta, [0, 0, 4, 2])
    assert scale == 8
    assert images[0][0, 1].tolist() == images[1][0, 1].tolist() == [128, 128, 128]
    assert images[0][1, 3, 2] == 159 and images[1][1, 3, 2] == 255
    assert images[0][0, 0, 0] < 128


def test_valid_crop_excludes_padding_from_the_shared_scale():
    A = np.ones((1, 3, 3, 5), dtype=np.float32)
    A[:, :, 2, :] = 100
    A[:, :, :, 4] = -200
    images, scale = signed_display(A, np.zeros_like(A), [0, 0, 4, 2])
    assert scale == 1 and images[0].shape == (2, 4, 3)
    assert (images[0] == 255).all() and (images[1] == 128).all()


def test_zero_scale_native_pngs_vector_titles_and_nonoverwrite(tmp_path):
    zero = np.zeros((1, 3, 8, 12), dtype=np.float32)
    images, scale = signed_display(zero, zero, [0, 0, 12, 8])
    assert scale == 0 and all((im == 128).all() for im in images)
    output = fresh_output(tmp_path / "anchor_residual_vis")
    render(images, output, 60)
    for stem in ("uicf_anchor_relative_compact", "uicf_correction_residual_compact"):
        with Image.open(output / f"{stem}.png") as im:
            assert im.size == (12, 8)
            np.testing.assert_array_equal(np.asarray(im), images[0])
    for stem in ("uicf_anchor_relative_signed", "uicf_correction_residual_signed"):
        assert '<text' in (output / f"{stem}.svg").read_text()
    newer = fresh_output(tmp_path / "anchor_residual_vis")
    assert newer != output and not any(newer.iterdir())
    assert (output / "uicf_anchor_residual_comparison.png").is_file()


def test_nonfinite_and_bad_shapes_are_rejected():
    I = np.zeros((1, 3, 2, 4), dtype=np.float32)
    with pytest.raises(ValueError, match="incompatible"):
        derive(I, np.zeros((3,), dtype=np.float16), I, I)
    bad = I.copy()
    bad.flat[0] = np.nan
    with pytest.raises(ValueError, match="finite"):
        derive(bad, np.zeros((1, 3), dtype=np.float16), I, I)
