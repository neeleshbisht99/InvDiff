import logging

import torch
from flask import Flask, jsonify, request
from transformers import AutoProcessor, AutoModelForVision2Seq
from PIL import Image

app = Flask(__name__)

# Setup logging
logging.basicConfig(level=logging.INFO)

device = torch.device("cuda:2") if torch.cuda.is_available() else "cpu"
logging.info("Loading model... This might take a while.")

vis_processors = AutoProcessor.from_pretrained("microsoft/kosmos-2-patch14-224")
model = AutoModelForVision2Seq.from_pretrained("microsoft/kosmos-2-patch14-224")
model = model.to(device).eval()
logging.info("Model loaded successfully!")

@app.route("/", methods=["POST"])
def interact_with_blip():
    if "image" not in request.files:
        return jsonify({"error": "Image not provided"}), 400

    if "text" not in request.form:
        return jsonify({"error": "Text not provided"}), 400

    raw_image = Image.open(request.files["image"])
    
    with torch.inference_mode():
        inputs = vis_processors(text=request.form["text"], images=raw_image, return_tensors="pt")
        inputs = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in inputs.items()}
        generated_ids = model.generate(
            pixel_values=inputs["pixel_values"],
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            image_embeds=None,
            image_embeds_position_mask=inputs["image_embeds_position_mask"],
            use_cache=True,
            max_new_tokens=32,
        )
        generated_text = vis_processors.batch_decode(generated_ids, skip_special_tokens=True)[0]
        processed_text, _ = vis_processors.post_process_generation(generated_text)
        result = processed_text.replace("Describe this image in one short sentence.", "")

    return jsonify({"input": request.form["text"], "output": result})


if __name__ == "__main__":
    logging.info("Server is running!")
    app.run(host="0.0.0.0", port=8094, debug=False)
