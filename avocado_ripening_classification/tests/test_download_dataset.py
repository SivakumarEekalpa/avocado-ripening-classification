import download_dataset


def test_default_dataset_download_warns_about_incompatible_labels(
    tmp_path,
    monkeypatch,
    capsys,
):
    commands = []
    monkeypatch.setattr(
        "sys.argv",
        ["download_dataset.py", "--output", str(tmp_path / "dataset.zip")],
    )
    monkeypatch.setattr(
        download_dataset.subprocess,
        "run",
        lambda command, check: commands.append((command, check)),
    )

    download_dataset.main()

    command, check = commands[0]
    assert download_dataset.DEFAULT_URL in command
    assert check is True
    assert "not the four ripeness classes" in capsys.readouterr().out


def test_custom_dataset_url_does_not_show_default_link_warning(
    tmp_path,
    monkeypatch,
    capsys,
):
    commands = []
    custom_url = "https://example.com/compatible-dataset.zip"
    monkeypatch.setattr(
        "sys.argv",
        [
            "download_dataset.py",
            "--url",
            custom_url,
            "--output",
            str(tmp_path / "dataset.zip"),
        ],
    )
    monkeypatch.setattr(
        download_dataset.subprocess,
        "run",
        lambda command, check: commands.append((command, check)),
    )

    download_dataset.main()

    assert custom_url in commands[0][0]
    assert "WARNING:" not in capsys.readouterr().out
