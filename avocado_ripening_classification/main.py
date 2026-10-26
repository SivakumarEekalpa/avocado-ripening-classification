"""Compare trained classical and MobileNetV2 models on one image.
Usage: python main.py --image path/to/avocado.jpg
Models are created only after running the training pipeline.
"""
import argparse, time
from pathlib import Path
import joblib, numpy as np, torch
import matplotlib.pyplot as plt
from src.feature_extraction import extract_features
from src.deep_learning_model import load_mobilenet
from src.utils import save_json
from PIL import Image
from torchvision import transforms

PROJECT_ROOT = Path(__file__).resolve().parent


def predict_classical(model_file,image_path):
    bundle=joblib.load(model_file)
    model=bundle["pipeline"]
    x=extract_features(image_path).reshape(1,-1)
    pred=model.predict(x)[0]
    proba=model.predict_proba(x)[0] if hasattr(model,"predict_proba") else None
    return pred, (float(np.max(proba)) if proba is not None else None)


def predict_dl(model_file,image_path):
    model,classes,device=load_mobilenet(model_file)
    tf=transforms.Compose([transforms.Resize((224,224)),transforms.ToTensor(),
                           transforms.Normalize([.485,.456,.406],[.229,.224,.225])])
    with Image.open(image_path) as image:
        x=tf(image.convert("RGB")).unsqueeze(0).to(device)
    if device.type == "cuda": torch.cuda.synchronize(device)
    start=time.perf_counter()
    with torch.inference_mode(): probs=torch.softmax(model(x),dim=1)[0]
    if device.type == "cuda": torch.cuda.synchronize(device)
    ms=(time.perf_counter()-start)*1000
    idx=int(probs.argmax().item())
    return classes[idx],float(probs[idx].item()),ms


def save_prediction_summary(image_path, predictions, output_path=None):
    if not predictions:
        return None

    if output_path is None:
        output_path = (
            PROJECT_ROOT
            / "outputs"
            / "predictions"
            / f"{Path(image_path).stem}_prediction.json"
        )
    else:
        output_path = Path(output_path)
        if not output_path.is_absolute():
            output_path = PROJECT_ROOT / output_path

    save_json(
        {
            "image": str(Path(image_path).resolve()),
            "predictions": predictions,
        },
        output_path,
    )
    return output_path


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--image",required=True)
    ap.add_argument("--models-dir",default="models")
    ap.add_argument(
        "--output",
        help="Prediction JSON path (default: outputs/predictions/<image>_prediction.json).",
    )
    ap.add_argument("--no-display",action="store_true")
    args=ap.parse_args(); p=Path(args.image)
    if not p.exists(): ap.error(f"Image not found: {p}")
    try:
        with Image.open(p) as image: image.verify()
    except (OSError, ValueError) as exc:
        ap.error(f"Image is unreadable: {p} ({exc})")
    md=Path(args.models_dir)
    print(f"Image: {p}")
    classical=[]
    predictions={}
    for name in ["svm","rf"]:
        f=md/f"{name}_model.joblib"
        if f.exists():
            pred,conf=predict_classical(f,p); classical.append((name,pred,conf))
            predictions[name]={"predicted_class":pred,"confidence":conf}
            print(f"{name.upper()}: {pred}"+(f" (confidence={conf:.3f})" if conf is not None else ""))
        else: print(f"{name.upper()}: model missing; run training first.")
    dl=md/"mobilenetv2.pth"
    title=" | ".join(f"{n}: {pred}" for n,pred,_ in classical)
    if dl.exists():
        pred,conf,ms=predict_dl(dl,p)
        predictions["mobilenetv2"]={
            "predicted_class":pred,
            "confidence":conf,
            "inference_ms":ms,
        }
        print(f"MobileNetV2: {pred} (probability={conf:.3f}, inference={ms:.2f} ms)")
        title += f" | MobileNetV2: {pred} ({conf:.2f})"
    else: print("MobileNetV2: model missing; run training first.")
    output_path=save_prediction_summary(p,predictions,args.output)
    if output_path is not None:
        print(f"Saved prediction summary: {output_path}")
    if not args.no_display:
        with Image.open(p) as image: img=np.asarray(image.convert("RGB"))
        plt.figure(figsize=(7,6)); plt.imshow(img); plt.axis("off"); plt.title(title or "No trained models found"); plt.tight_layout(); plt.show()

if __name__=="__main__": main()
