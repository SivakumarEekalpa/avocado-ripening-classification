import pandas as pd
import pytest

from src.classical_model import train_classical


def test_classical_training_rejects_rare_cv_classes_before_feature_work():
    training = pd.DataFrame(
        {
            "path": [f"{label}.png" for label in ("underripe", "breaking", "ripe", "overripe")],
            "label": ["underripe", "breaking", "ripe", "overripe"],
        }
    )
    validation = training.copy()
    test = training.copy()

    with pytest.raises(ValueError, match="at least 4 training images per class"):
        train_classical(training, validation, test)


def test_classical_training_rejects_empty_splits():
    training = pd.DataFrame({"path": ["image.png"], "label": ["ripe"]})
    empty = pd.DataFrame(columns=["path", "label"])

    with pytest.raises(ValueError, match="validation split cannot be empty"):
        train_classical(training, empty, training)
