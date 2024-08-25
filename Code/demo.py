import torch
import model_loader
import pipeline
from PIL import Image
from transformers import CLIPTokenizer

DEVICE = "mps"
print(f"Using device: {DEVICE}")

tokenizer = CLIPTokenizer("data/tokenizer_vocab.json", merges_file="data/tokenizer_merges.txt")
model_path = "data/v1-5-pruned-emaonly.ckpt"
model_components = model_loader.load_model_weights(model_path, DEVICE)

prompt = "A cat stretching on the floor, highly detailed, ultra sharp, cinematic, 100mm lens, 8k resolution." # need more refined details
uc_prompt = ""
use_cfg = True
cfg_scale = 10 # 9-11 seems to work best
sampler = "euler a"
num_inference_steps = 50
seed = 13

output_image = pipeline.generate(
    prompt=prompt,
    uc_prompt=uc_prompt,
    use_cfg=use_cfg,
    cfg_scale=cfg_scale,
    sampler_name=sampler,
    num_inference_steps=num_inference_steps,
    models=model_components,
    seed=seed,
    device=DEVICE,
    idle_device="cpu",
    tokenizer=tokenizer,
)

output_path = "output_imgs/img_101.png"
output_img = Image.fromarray(output_image)
output_img.save(output_path)