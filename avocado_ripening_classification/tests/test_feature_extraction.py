import numpy as np
from PIL import Image

from src.feature_extraction import extract_features


def test_feature_vector_has_finite_expected_shape_and_normalized_hue_histogram(tmp_path):
    pixels = np.zeros((80, 96, 3), dtype=np.uint8)
    pixels[:, :48] = (20, 180, 50)
    pixels[:, 48:] = (180, 50, 90)
    image_path = tmp_path / "two_color_avocado.png"
    Image.fromarray(pixels).save(image_path)

    features = extract_features(image_path)

    assert features.shape == (206,)
    assert np.isfinite(features).all()
    # RGB contributes 48 values; the first HSV channel is hue with its [0, 180) range.
    assert np.isclose(features[48:64].sum(), 1.0, atol=1e-6)
