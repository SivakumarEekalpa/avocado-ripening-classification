from train import PROJECT_ROOT, _project_path


def test_relative_training_paths_are_anchored_to_project_root():
    assert _project_path("data/raw") == PROJECT_ROOT / "data" / "raw"


def test_absolute_training_paths_are_preserved(tmp_path):
    absolute_path = tmp_path / "dataset"

    assert _project_path(absolute_path) == absolute_path
