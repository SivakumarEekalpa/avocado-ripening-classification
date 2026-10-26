"""Download the assignment's Google Drive file using gdown; inspect archive before extraction."""
import argparse, subprocess, sys
from pathlib import Path

DEFAULT_URL = "https://drive.google.com/file/d/1NMJ0FOxcWH1mwFFeERuY6fgEY-MVgeET/view"


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--url",default=DEFAULT_URL)
    p.add_argument("--output",default="data/avocado_dataset_download")
    args=p.parse_args(); out=Path(args.output); out.parent.mkdir(parents=True,exist_ok=True)
    cmd=[sys.executable,"-m","gdown","--fuzzy",args.url,"-O",str(out)]
    subprocess.run(cmd,check=True)
    print(
        f"Downloaded to {out}. Inspect class folders before extracting anything "
        "into data/raw/."
    )
    if args.url == DEFAULT_URL:
        print(
            "WARNING: The configured Drive file currently contains "
            "blackspot/canker/fresh/grenning classes, not the four ripeness "
            "classes. Do not use it for ripeness training."
        )
if __name__=="__main__": main()
