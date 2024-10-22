# Flux for 8GB of VRAM

from diffusers import FluxPipeline, FluxTransformer2DModel
from transformers import T5EncoderModel
import torch
import gc
from huggingface_hub import login
import os
from dotenv import load_dotenv

load_dotenv()
login(token=os.getenv("HF_TOKEN"))

def flush():
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.reset_max_memory_allocated()
    torch.cuda.reset_peak_memory_stats()


def bytes_to_giga_bytes(bytes):
    return bytes / 1024 / 1024 / 1024


flush()

ckpt_id = "black-forest-labs/FLUX.1-dev"
ckpt_4bit_id = "sayakpaul/flux.1-dev-nf4-pkg"
prompt = "1girl, full body, squatting, solo, short hair, brown hair, gray eyes, headband, face paint, earrings, white sleeveless top, navy under dress, arm cuffs, fur headdress, fur cape, light-skinned female, female focus, looking at viewer, smiling, forest"


text_encoder_2_4bit = T5EncoderModel.from_pretrained(
    ckpt_4bit_id,
    subfolder="text_encoder_2",
)

pipeline = FluxPipeline.from_pretrained(
    ckpt_id,
    text_encoder_2=text_encoder_2_4bit,
    transformer=None,
    vae=None,
    torch_dtype=torch.float16,
)
pipeline.enable_model_cpu_offload()


with torch.no_grad():
    print("Encoding prompts.")
    prompt_embeds, pooled_prompt_embeds, text_ids = pipeline.encode_prompt(
        prompt=prompt, prompt_2=None, max_sequence_length=256
    )


pipeline = pipeline.to("cpu")
del pipeline

flush()


transformer_4bit = FluxTransformer2DModel.from_pretrained(ckpt_4bit_id, subfolder="transformer",
    low_cpu_mem_usage=False,
    ignore_mismatched_sizes=True)
pipeline = FluxPipeline.from_pretrained(
    ckpt_id,
    text_encoder=None,
    text_encoder_2=None,
    tokenizer=None,
    tokenizer_2=None,
    transformer=transformer_4bit,
    torch_dtype=torch.float16,
)
pipeline.enable_model_cpu_offload()

print("Running denoising.")
height, width = 512, 768
images = pipeline(
    prompt_embeds=prompt_embeds,
    pooled_prompt_embeds=pooled_prompt_embeds,
    num_inference_steps=50,
    guidance_scale=5.5,
    height=height,
    width=width,
    output_type="pil",
).images
images[0].save("output.png")