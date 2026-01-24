import logging
import re

import torch
from flask import Flask, jsonify, request
from PIL import Image
from transformers import BlipForConditionalGeneration, BlipProcessor

app = Flask(__name__)

logging.basicConfig(level=logging.INFO)

device = torch.device("cuda:2") if torch.cuda.is_available() else "cpu"
dtype = torch.float16 if torch.cuda.is_available() else torch.float32

MODEL_ID = "adibvafa/BLIP-MIMIC-CXR"

logging.info("Loading BLIP-MIMIC-CXR model... This might take a while.")
processor = BlipProcessor.from_pretrained(MODEL_ID)
model = BlipForConditionalGeneration.from_pretrained(MODEL_ID)
model.to(device)
model.eval()
logging.info("Model loaded successfully!")


@app.route("/", methods=["POST"])
def interact_with_blip():
    if "image" not in request.files:
        return jsonify({"error": "Image not provided"}), 400

    if "text" not in request.form:
        return jsonify({"error": "Text not provided"}), 400

    prompt = request.form["text"]

    raw_image = Image.open(request.files["image"]).convert("RGB")

    with torch.no_grad():
        inputs = processor(
            images=raw_image,
            text=prompt,
            return_tensors="pt",
        )
        # Move tensors to device
        inputs = {k: v.to(device) for k, v in inputs.items()}

        output_ids = model.generate(
            **inputs,
            max_new_tokens=150,   # give it room
            do_sample=False,
            num_beams=1,
            early_stopping=True,
        )

        input_len = inputs["input_ids"].shape[1]
        gen_only_ids = output_ids[0][input_len:]
        result = processor.tokenizer.decode(gen_only_ids, skip_special_tokens=True).strip()
        result = re.sub(r"_+", "", result).strip(" .:-")
    return jsonify({"input": prompt, "output": result})


if __name__ == "__main__":
    logging.info("Server is running!")
    app.run(host="0.0.0.0", port=8092, debug=False)