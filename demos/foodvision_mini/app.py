import gradio as gr
import os
import torch

from model import create_effnetb2_model
from timeit import default_timer as timer
from typing import Tuple, Dict

class_names = ["pizza", "steak", "sushi"]

# Absolute path to the model weights file
weights_path = "./pretrained_effnetb2_feature_extractor_pizza_steak_sushi_20_percent.pth"

# Check if the file exists
if not os.path.exists(weights_path):
    raise FileNotFoundError(f"Model weights file not found at: {weights_path}")

# Create EffNetB2 model
effnetb2, effnetb2_transforms = create_effnetb2_model(
    num_classes=len(class_names),  # number of classes in the dataset
)

# Load saved weights
effnetb2.load_state_dict(
    torch.load(weights_path, map_location=torch.device("cpu"))
)

# Predict function 
def predict(img) -> Tuple[Dict, float]:
    start_time = timer()
    img = effnetb2_transforms(img).unsqueeze(0)
    effnetb2.eval()
    with torch.inference_mode():
        pred_probs = torch.softmax(effnetb2(img), dim=1)
    
    pred_labels_and_probs = {class_names[i]: float(pred_probs[0][i]) for i in range(len(class_names))}
    pred_time = round(timer() - start_time, 5)
    
    return pred_labels_and_probs, pred_time

# Gradio app configuration
title = "FoodVision Mini 🍕🥩🍣"
description = "An EfficientNetB2 feature extractor computer vision model to classify images of food as pizza, steak or sushi."
article = "Created at [PyTorch Model Deployment]"

example_list = [["examples/" + example] for example in os.listdir("examples")]

# Create the Gradio demo
demo = gr.Interface(fn=predict,
                    inputs=gr.Image(type="pil"),
                    outputs=[gr.Label(num_top_classes=3, label="Predictions"),
                             gr.Number(label="Prediction time (s)")],
                    examples=example_list,
                    title=title,
                    description=description,
                    article=article)

# Launch the demo
demo.launch()
