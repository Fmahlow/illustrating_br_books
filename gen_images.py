from diffusers import AutoPipelineForText2Image, StableDiffusionXLPipeline, StableDiffusionXLImg2ImgPipeline, DiffusionPipeline
import torch
from PIL import Image

base = DiffusionPipeline.from_pretrained(
    "stabilityai/stable-diffusion-xl-base-1.0", torch_dtype=torch.float16, variant="fp16", use_safetensors=True
)
base.to("cuda")

refiner = DiffusionPipeline.from_pretrained(
    "stabilityai/stable-diffusion-xl-refiner-1.0",
    text_encoder_2=base.text_encoder_2,
    vae=base.vae,
    torch_dtype=torch.float16,
    use_safetensors=True,
    variant="fp16",
)
refiner.to("cuda")

lista_de_prompts = ["A painting of Brazilian caatinga as a lush valley, with a vast orchard. Vibrant hues of the trees contrast with the arid landscape around, digital art", 
                    "A painting of Brazilian caatinga as a dry land, some cracks in the soil, leafless trees and cactus. The sun shines in the sky, digital art",
                    "A painting of Brazilian caatinga as a lush valley, yet with some dry land aspects. Animals from caatinga in the background (tapir, suçuarana, seriemas). Digital art",
                    "A painting of a man from the Brazilian backlands. He is a determined person, yet his countenance reflects the harsh life in the backlands, 19th century, realis",
                    "A painting of a Brazilian army expedition in the arid Brazilian caatinga, with calvary and war cannons, 19th century, realism"]

num_samples = 300
for i in lista_de_prompts:
    for j in range(1, num_samples+1):
        print(j)
        image = base(
        prompt=i,
        num_inference_steps=40,
        denoising_end=0.8,
        output_type="latent",
        negative_prompt="bad anatomy, ugly face, extra fingers, ugly hands, ugly face, extra legs, bad legs, wearing robes").images
        image = refiner(
            prompt=i,
            num_inference_steps=40,
            denoising_start=0.8,
            image=image,
        ).images[0]
        image.save(f"imgs_geradas_sertoes/{i[0:200]}_{j}.jpg")