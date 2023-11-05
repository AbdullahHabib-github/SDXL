from diffusers import DiffusionPipeline
import torch
from compel import Compel, ReturnedEmbeddingsType
from diffusers import DiffusionPipeline, AutoencoderKL
from tqdm.auto import tqdm
from random import randint
from PIL import Image

class Model():
    def __init__(self):
        print("hehahehahhfaef")
        torch.cuda.empty_cache()

        vae = AutoencoderKL.from_pretrained("madebyollin/sdxl-vae-fp16-fix",
                                            torch_dtype=torch.float16)
        self.base = DiffusionPipeline.from_pretrained(
            "stabilityai/stable-diffusion-xl-base-1.0",
            vae=vae,
            torch_dtype=torch.float16,
            variant="fp16",
            use_safetensors=True
        )

        self.base.load_lora_weights("minimaxir/sdxl-wrong-lora")

        _ = self.base.to("cuda")
        # base.enable_model_cpu_offload()  # recommended for T4 GPU if enough system RAM

        self.refiner = DiffusionPipeline.from_pretrained(
            "stabilityai/stable-diffusion-xl-refiner-1.0",
            text_encoder_2=self.base.text_encoder_2,
            vae=self.base.vae,
            torch_dtype=torch.float16,
            use_safetensors=True,
            variant="fp16",
        )

        _ = self.refiner.to("cuda")



        self.compel_base = Compel(tokenizer=[self.base.tokenizer, self.base.tokenizer_2] , text_encoder=[self.base.text_encoder, self.base.text_encoder_2], returned_embeddings_type=ReturnedEmbeddingsType.PENULTIMATE_HIDDEN_STATES_NON_NORMALIZED, requires_pooled=[False, True])
        self.compel_refiner = Compel(tokenizer=self.refiner.tokenizer_2 , text_encoder=self.refiner.text_encoder_2, returned_embeddings_type=ReturnedEmbeddingsType.PENULTIMATE_HIDDEN_STATES_NON_NORMALIZED, requires_pooled=True)
        torch.cuda.empty_cache()



    def gen_image(self,source_prompt, negative_prompt,cfg=13, seed=-1, webp_output=True):
        if seed < 0:
            seed = randint(0, 10**8)
            print(f"Seed: {seed}")
        # return Image.open("8f2563f8-d71c-4ea5-b20f-c4493c8b382a.jpeg")
        torch.cuda.empty_cache()

        high_noise_frac = 0.8
        prompt = source_prompt
        negative_prompt = "wrong"  # hardcoding

        conditioning, pooled = self.compel_base(prompt)
        conditioning_neg, pooled_neg = self.compel_base(negative_prompt) if negative_prompt is not None else (None, None)
        generator = torch.Generator(device="cuda").manual_seed(seed)

        latents = self.base(prompt_embeds=conditioning,
                    pooled_prompt_embeds=pooled,
                    negative_prompt_embeds=conditioning_neg,
                    negative_pooled_prompt_embeds=pooled_neg,
                    guidance_scale=cfg,
                    denoising_end=high_noise_frac,
                    generator=generator,
                    output_type="latent",
                    cross_attention_kwargs={"scale": 1.}
                    ).images

        conditioning, pooled = self.compel_refiner(prompt)
        conditioning_neg, pooled_neg = self.compel_refiner(negative_prompt) if negative_prompt is not None else (None, None)
        generator = torch.Generator(device="cuda").manual_seed(seed)

        images = self.refiner(
            prompt_embeds=conditioning,
            pooled_prompt_embeds=pooled,
            negative_prompt_embeds=conditioning_neg,
            negative_pooled_prompt_embeds=pooled_neg,
            guidance_scale=cfg,
            denoising_start=high_noise_frac,
            image=latents,
            generator=generator,
            ).images
        
        torch.cuda.empty_cache()

        image = images[0]

        img="img.png"

        if webp_output:
            img = "img.webp"
            image.save("img.webp", format="webp")
        else:
            image.save("img.png")

        return Image.open(img)