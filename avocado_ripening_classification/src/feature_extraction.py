"""Color histograms, LAB statistics, LBP, and GLCM features for classical ML."""
import cv2
import numpy as np
from skimage.feature import local_binary_pattern, graycomatrix, graycoprops
from .preprocessing import read_bgr


def extract_features(image_path, size=(128,128)):
    bgr = read_bgr(image_path, size=size, denoise=True)
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)
    feats = []
    # OpenCV represents HSV hue in [0, 180), while the other channels use [0, 256).
    for img, channel_ranges in (
        (rgb, [(0, 256)] * 3),
        (hsv, [(0, 180), (0, 256), (0, 256)]),
        (lab, [(0, 256)] * 3),
    ):
        for channel, value_range in enumerate(channel_ranges):
            hist = cv2.calcHist([img], [channel], None, [16], list(value_range)).flatten()
            hist = hist / (hist.sum() + 1e-8)
            feats.extend(hist.tolist())
    # Explicit Lab a* statistics: useful but not sufficient alone to infer ripeness.
    a = lab[:,:,1].astype(np.float32)
    feats.extend([float(a.mean()/255), float(a.std()/255), float(np.percentile(a,10)/255), float(np.percentile(a,90)/255)])
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    lbp = local_binary_pattern(gray, P=8, R=1, method="uniform")
    hist, _ = np.histogram(lbp.ravel(), bins=np.arange(0, 11), range=(0,10), density=True)
    feats.extend(np.nan_to_num(hist).tolist())
    # Quantize gray image for GLCM.
    quant = (gray // 16).astype(np.uint8)
    glcm = graycomatrix(quant, distances=[1,3], angles=[0,np.pi/4,np.pi/2,3*np.pi/4],
                        levels=16, symmetric=True, normed=True)
    for prop in ["contrast","dissimilarity","homogeneity","energy","correlation","ASM"]:
        feats.extend(graycoprops(glcm, prop).flatten().tolist())
    return np.asarray(feats, dtype=np.float32)
