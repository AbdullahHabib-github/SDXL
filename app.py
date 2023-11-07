from sdxl import Model
import torch
import streamlit as st

torch.cuda.empty_cache()
@st.cache_resource
def calling_the_genie():
    return Model()
generator = calling_the_genie()

torch.cuda.empty_cache()

# @st.cache_data
def app():
    st.title("Cinas Photo Ginie")
    torch.cuda.empty_cache()

    allocated_memory = torch.cuda.memory_allocated()
    cached_memory = torch.cuda.memory_reserved()

    print("allocated_memory",allocated_memory)
    print("cached_memory",cached_memory)

    # Input parameters using Streamlit widgets
    prompt = st.text_area("Prompt", "A realistic High-Quality photo of outdoor area in a stylish Scandinavian vacation bungalow, Black house, Forest, sunlight, wooden porch")
    neg_prompt = st.text_area("Negative Prompt", "wrong,furniture, unnatural lighting")
    cfg = st.slider("CFG", min_value=7.0, max_value=15.0, step=0.5, value=15.0)
    high_noise_frac = st.slider("High Noise Frac", min_value=0.0, max_value=1.0, step=0.05, value=0.8)
    seed = st.number_input("Seed: Set seed less than 0 to let the model choose a seed randomly.", value=-1, step=1)
    webp_output = False


    # Generate image button
    if st.button("Generate Image"):
        # Call your image generation function
        message_placeholder = st.empty()
        message_placeholder.text("Wait while I do the magic...")
        torch.cuda.empty_cache()

        generated_image = generator.gen_image(prompt, neg_prompt,high_noise_frac, cfg, seed, webp_output)
        torch.cuda.empty_cache()
        allocated_memory = torch.cuda.memory_allocated()
        cached_memory = torch.cuda.memory_reserved()

        print("lolallocated_memory", allocated_memory)
        print("lolcached_memory", cached_memory)
        message_placeholder.text("Voila!")
        st.text(f"Seed set to {generated_image[1]}")

        # Display the generated image
        st.image(generated_image[0], caption="Generated Image", use_column_width=True)

# torch.cuda.empty_cache()
app()