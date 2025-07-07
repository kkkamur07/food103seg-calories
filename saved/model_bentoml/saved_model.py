import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))
import bentoml
import torch
from src.segmentation.model import MiniUNet

model = MiniUNet()

#Loading the model weights
dict=torch.load("saved/models/best_model.pth")
model.load_state_dict(dict["model_state_dict"])
model.eval()

# Save the model from file path
bentoml.pytorch.save_model(
    "food_segmentation_model",
    model,
    signatures={"__call__": {"batchable": False}},
    )
