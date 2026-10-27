import time
import torch
from thop import profile as thop_profile
from vision_models import StackedDenseClassifier, SpatialConvClassifier, PatchVisionTransformer

print("[INFO] Miray Buyuktepe - Lab 03 Vision Architecture Benchmark Suite Ready.")
