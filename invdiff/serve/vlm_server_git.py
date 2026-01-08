import logging

import torch
from flask import Flask, jsonify, request
from transformers import AutoProcessor, AutoModelForCausalLM
from PIL import Image

app = Flask(__name__)

# Setup logging
logging.basicConfig(level=logging.INFO)

device = torch.device("cuda:2") if torch.cuda.is_available() else "cpu"
logging.info("Loading model... This might take a while.")

vis_processors = AutoProcessor.from_pretrained("microsoft/git-base-coco")
model = AutoModelForCausalLM.from_pretrained("microsoft/git-base-coco")
logging.info("Model loaded successfully!")

@app.route("/", methods=["POST"])
def interact_with_blip():
    if "image" not in request.files:
        return jsonify({"error": "Image not provided"}), 400

    if "text" not in request.form:
        return jsonify({"error": "Text not provided"}), 400

    raw_image = Image.open(request.files["image"]).convert("RGB")
    
    with torch.no_grad():
        pixel_values = vis_processors(images=raw_image, return_tensors="pt").pixel_values
        generated_ids = model.generate(pixel_values=pixel_values, max_length=50)
        result = vis_processors.batch_decode(generated_ids, skip_special_tokens=True)[0]

    return jsonify({"input": request.form["text"], "output": result})


if __name__ == "__main__":
    logging.info("Server is running!")
    app.run(host="0.0.0.0", port=8092, debug=False)
