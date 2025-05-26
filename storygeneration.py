import streamlit as st
import textwrap
from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM

# Set page config first
st.set_page_config(
    page_title="Story Generator",
    page_icon=":book:",
    layout="wide"
)

# Load the tokenizer and the trained model from local path
path = 'D:/Udemy MLPROJECT/streamlit/gpt2-story-gen'

@st.cache_resource
def load_model():
    tokenizer = AutoTokenizer.from_pretrained(path)
    model = AutoModelForCausalLM.from_pretrained(path)
    tokenizer.pad_token = tokenizer.eos_token
    return tokenizer, model

st.title("Story Generation")
st.markdown("Give your imagination wings! Enter a prompt and generate creative stories.")

tokenizer, model = load_model()

generator = pipeline(
    'text-generation',
    model=model,  # use the model object, not the path
    tokenizer=tokenizer,
    clean_up_tokenization_spaces=True
)

# Add a sidebar for user input
st.sidebar.title("Story Input")
st.sidebar.write("Enter a prompt to generate a story:")
prompt = st.sidebar.text_area(
    "Prompt",
    placeholder="Type your story prompt here...",
    height=200
)

story_1 = False

if st.sidebar.button("Generate Story"):
    if not prompt.strip():
        st.sidebar.error("❗Please enter a prompt before generating a story.")
    else:
        with st.spinner("Generating your story... 🚀"):
            outputs = generator(
                prompt,
                max_length=750,
                num_return_sequences=1,
                do_sample=True,
                top_p=0.95,
                temperature=0.9,
                repetition_penalty=1.2,
                truncation=True
            )
        
        story = outputs[0]['generated_text'].replace('\n', ' ').strip()
        story = textwrap.fill(story, width=100)
        st.success("Story generated successfully!")
        st.markdown(
            f"""
            <div style="background-color:#f0f2f6;padding:20px;border-radius:10px;margin-bottom:20px;">
                <h4 style="color:#4B8BBE;">Generated Story</h4>
                <p style="text-align:justify;font-size:17px;color:#333;">{story}</p>
            </div>
            """,
            unsafe_allow_html=True
        )
        story_1 = True


else:
    st.write("Enter a prompt and click 'Generate Story' to create a story.")

# If the story has been generated, allow the user to generate more stories
if story_1:
    st.sidebar.write("You can generate more stories based on the same prompt.")
# Button to generate more stories based on the same prompt

if st.button("Generate more stories", key="generate_more"):
    if not prompt.strip():
        st.sidebar.error("❗Please enter a prompt before generating more stories.")
    else:
        with st.spinner("Generating more stories... 🚀"):
            # Generate two more stories based on the same prompt
            outputs = generator(
                prompt,
                max_length=750,
                num_return_sequences=2,
                do_sample=True,
                top_p=0.95,
                temperature=0.9,
                repetition_penalty=1.2,
                truncation=True
            )

        st.subheader("Generated Stories")
        for i, output in enumerate(outputs):
            story = output['generated_text'].replace('\n', ' ').strip()
            story = textwrap.fill(story, width=100)
            st.markdown(
                f"""
                <div style="background-color:#f9f9f9;padding:20px;border-radius:10px;margin-bottom:20px;">
                    <h5 style="color:#0b5ed7;">Story {i + 1}</h5>
                    <p style="text-align:justify;font-size:17px;color:#222;">{story}</p>
                </div>
                """,
                unsafe_allow_html=True
            )
        st.toast("You can modify the prompt and generate new stories as needed.")

