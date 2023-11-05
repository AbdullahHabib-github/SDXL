from diffusers import DiffusionPipeline, AutoencoderKL
from compel import Compel, ReturnedEmbeddingsType
from PIL import Image
import streamlit as st
import torch
from random import randint

def load_models():
    vae = AutoencoderKL.from_pretrained("madebyollin/sdxl-vae-fp16-fix", torch_dtype=torch.float16)
    
    base = DiffusionPipeline.from_pretrained(
        "stabilityai/stable-diffusion-xl-base-1.0",
        vae=vae,
        torch_dtype=torch.float16,
        variant="fp16",
        use_safetensors=True
    )
    
    base.load_lora_weights("minimaxir/sdxl-wrong-lora")
    _ = base.to("cuda")

    refiner = DiffusionPipeline.from_pretrained(
        "stabilityai/stable-diffusion-xl-refiner-1.0",
        text_encoder_2=base.text_encoder_2,
        vae=base.vae,
        torch_dtype=torch.float16,
        use_safetensors=True,
        variant="fp16",
    )
    
    _ = refiner.to("cuda")

    compel_base = Compel(
        tokenizer=[base.tokenizer, base.tokenizer_2],
        text_encoder=[base.text_encoder, base.text_encoder_2],
        returned_embeddings_type=ReturnedEmbeddingsType.PENULTIMATE_HIDDEN_STATES_NON_NORMALIZED,
        requires_pooled=[False, True]
    )
    
    compel_refiner = Compel(
        tokenizer=refiner.tokenizer_2,
        text_encoder=refiner.text_encoder_2,
        returned_embeddings_type=ReturnedEmbeddingsType.PENULTIMATE_HIDDEN_STATES_NON_NORMALIZED,
        requires_pooled=True
    )

    return base, refiner, compel_base, compel_refiner

def gen_image(base, refiner, compel_base, compel_refiner, source_prompt, negative_prompt,high_noise_frac = 0.8, cfg=13, seed=-1, webp_output=False):

    if seed < 0:
        seed = randint(0, 10**8)
        print(f"Seed: {seed}")

    prompt = source_prompt
    # negative_prompt = "wrong,furniture, unnatural lighting"  # hardcoding

    conditioning, pooled = compel_base(prompt)
    conditioning_neg, pooled_neg = compel_base(negative_prompt) if negative_prompt is not None else (None, None)
    generator = torch.Generator(device="cuda").manual_seed(seed)

    latents = base(prompt_embeds=conditioning,
                pooled_prompt_embeds=pooled,
                negative_prompt_embeds=conditioning_neg,
                negative_pooled_prompt_embeds=pooled_neg,
                guidance_scale=cfg,
                denoising_end=high_noise_frac,
                generator=generator,
                output_type="latent",
                cross_attention_kwargs={"scale": 1.}
                ).images

    conditioning, pooled = compel_refiner(prompt)
    conditioning_neg, pooled_neg = compel_refiner(negative_prompt) if negative_prompt is not None else (None, None)
    generator = torch.Generator(device="cuda").manual_seed(seed)

    images = refiner(
        prompt_embeds=conditioning,
        pooled_prompt_embeds=pooled,
        negative_prompt_embeds=conditioning_neg,
        negative_pooled_prompt_embeds=pooled_neg,
        guidance_scale=cfg,
        denoising_start=high_noise_frac,
        image=latents,
        generator=generator,
        ).images

    image = images[0]

    # display(image.resize((image.width // 2, image.height // 2)))
    if webp_output:
        image.save("img.webp", format="webp")
        return Image.open("img.webp")
    else:
        image.save("img.png")
        return Image.open("img.png")

def main():
    st.title("Image Generation App")

    # Load models only when the "Generate Image" button is clicked
    if "models_loaded" not in st.session_state:
        st.session_state.models_loaded = True
        base, refiner, compel_base, compel_refiner = load_models()
    else:
        base, refiner, compel_base, compel_refiner = st.session_state.base, st.session_state.refiner, st.session_state.compel_base, st.session_state.compel_refiner


    # Input parameters using Streamlit widgets
    prompt = st.text_area("Prompt", "A realistic High-Quality photo of outdoor area in a stylish Scandinavian vacation bungalow, Black house, Forest, sunlight, wooden porch")
    neg_prompt = st.text_area("Negative Prompt", "wrong,furniture, unnatural lighting")
    cfg = st.slider("CFG", min_value=7.0, max_value=15.0, step=0.5, value=15.0)
    high_noise_frac = st.slider("High Noise Factor", min_value=0.0, max_value=1.0, step=0.05, value=0.8)
    seed = st.number_input("Seed", value=-1, step=1)
    webp_output = st.checkbox("WebP Output")

    # Generate image button
    if st.button("Generate Image"):
        # Call your image generation function
        generated_image = gen_image(base, refiner, compel_base, compel_refiner,prompt,neg_prompt,high_noise_frac, cfg, seed, webp_output)

        # Display the generated image
        st.image(generated_image, caption="Generated Image", use_column_width=True)

if __name__ == "__main__":
    main()
