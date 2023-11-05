from mo import Model
import torch
import streamlit as st

torch.cuda.empty_cache()
generator = Model()



def app():
    st.title("Cinas Photo Ginie")
    torch.cuda.empty_cache()

    generator = Model()

    allocated_memory = torch.cuda.memory_allocated()
    cached_memory = torch.cuda.memory_reserved()

    print("allocated_memory",allocated_memory)
    print("cached_memory",cached_memory)

    # Input parameters using Streamlit widgets
    prompt = st.text_area("Prompt", "A realistic High-Quality photo of outdoor area in a stylish Scandinavian vacation bungalow, Black house, Forest, sunlight, wooden porch")
    neg_prompt = st.text_area("Negative Prompt", "wrong,furniture, unnatural lighting")
    cfg = st.slider("CFG", min_value=7.0, max_value=15.0, step=0.5, value=15.0)
    seed = st.number_input("Seed", value=-1, step=1)
    webp_output = False


    # Generate image button
    if st.button("Generate Image"):
        # Call your image generation function
        torch.cuda.empty_cache()
        generated_image = generator.gen_image(prompt, neg_prompt, cfg, seed, webp_output)
        torch.cuda.empty_cache()
        allocated_memory = torch.cuda.memory_allocated()
        cached_memory = torch.cuda.memory_reserved()

        print("lolallocated_memory", allocated_memory)
        print("lolcached_memory", cached_memory)
        # Display the generated image
        st.image(generated_image, caption="Generated Image", use_column_width=True)

# torch.cuda.empty_cache()
app()