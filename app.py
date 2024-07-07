import os
import pandas as pd
from PIL import Image
import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from langchain_community.llms import Ollama

# Cache the model and vectorizer to avoid reloading
@st.cache_resource
def load_llm_model():
    return Ollama(model="llama3")

@st.cache_resource
def load_vectorizer(df):
    vectorizer = TfidfVectorizer().fit(df['Prompt'])
    prompt_vectors = vectorizer.transform(df['Prompt'])
    return vectorizer, prompt_vectors

# Cache the image dictionary
@st.cache_data
def create_image_dict(base_path, df):
    image_dict = {}
    for _, row in df.iterrows():
        img_path = os.path.join(base_path, row['Image Name'])
        with Image.open(img_path) as img:
            img = img.resize((150, 150), Image.LANCZOS)  # Resize image to save memory
            image_dict[row['Image Name']] = img
    return image_dict

# Function to match user prompt
def match_prompt(user_prompt, df, vectorizer, prompt_vectors, num_images=3):
    user_prompt_vector = vectorizer.transform([user_prompt])
    similarities = cosine_similarity(user_prompt_vector, prompt_vectors)
    match_indices = similarities.argsort()[0][-num_images:][::-1]  # Get top 'num_images' matches
    matched_rows = df.iloc[match_indices]
    matched_rows['Similarity'] = similarities[0, match_indices]  # Add similarity scores to the DataFrame
    matched_rows = matched_rows.sort_values(by='Similarity', ascending=False)  # Sort by similarity
    return matched_rows

# Function to display images in Streamlit
def display_images(llm, image_dict, matched_rows):
    st.subheader("Top Matches:")
    for _, row in matched_rows.iterrows():
        image = image_dict[row['Image Name']]
        image_name = row['Image Name']
        
        col1, col2 = st.columns([1, 3])  # Create two columns for image and description
        
        with col1:
            st.image(image, use_column_width=True)  # Display the image
        
        with col2:
            prompt = (
                f"Now you are a Fashion Expert. I will be giving the image short description of an outfit, provide a description of what occasion that outfit can be used, then what kind of shoes and watches to be used for that. The image description is '{image_name}'. Dont print the image name in the output "
            )
            
            # Refine the query using LLM
            try:
                refined_disp = llm.invoke(prompt)
                st.write(refined_disp)  # Display the refined description
            except Exception as e:
                st.error(f"Error: {e}")

# Main function to run the Streamlit app
def main():
    st.set_page_config(page_title="StyleSage", page_icon=":dress:", layout="wide")
    
    # Inject CSS for custom styling
    st.markdown("""
        <style>
        body {
            font-family: 'Helvetica', sans-serif;
        }
        .stButton button {
            background-color: #4CAF50; /* Green */
            border: none;
            color: white;
            padding: 10px 20px;
            text-align: center;
            text-decoration: none;
            display: inline-block;
            font-size: 16px;
            margin: 10px 5px;
            cursor: pointer;
            border-radius: 12px;
            transition: background-color 0.3s ease;
            width: 150px; /* Fixed width for both buttons */
        }
        .stButton button:hover {
            background-color: white;
            color: #4CAF50;
            border: 2px solid #4CAF50;
        }
        .stButton button:active {
            background-color: #4CAF50;
            transform: translateY(1px);
        }
        .title h1 {
            font-size: 3em;
            color: #4CAF50;
        }
        .sidebar .sidebar-content h1 {
            font-size: 2em;
            color: #4CAF50;
        }
        .sidebar .sidebar-content h3 {
            font-size: 1.5em;
            color: #4CAF50;
        }
        .sidebar .sidebar-content {
            font-size: 1.2em;
        }
        .sidebar {
            background-color: #f0f0f0;
            background-image: linear-gradient(45deg, #f3ec78, #af4261);
            padding: 20px;
        }
        </style>
        """, unsafe_allow_html=True)
    
    st.sidebar.markdown("<h1>StyleSage👗</h1>", unsafe_allow_html=True)
    st.sidebar.markdown("<h3>Elevate Your Wardrobe with StyleSage!</h3>", unsafe_allow_html=True)
    st.sidebar.markdown("""
        <h2>Introducing Style Sage: Your Ultimate Fashion Companion</h2>

        <p>In the fast-paced world of fashion, keeping up with the latest trends, understanding what suits you best, and effortlessly combining pieces to create stunning outfits can be overwhelming. Enter Style Sage, your personalized fashion consultant available at your fingertips 24/7. Whether you're a fashion novice or a seasoned style enthusiast, Style Sage is designed to help you navigate the intricate landscape of fashion with ease and confidence.</p>

        <h3>What is Style Sage?</h3>

        <p>Style Sage is an innovative chatbot powered by advanced AI technology, crafted specifically to cater to your fashion needs. It’s more than just a virtual assistant; it's your personal stylist, trend advisor, and shopping guide all rolled into one. With a deep understanding of fashion trends, body types, personal preferences, and the latest in retail offerings, Style Sage provides tailored recommendations to elevate your style and transform your wardrobe.</p>

        <h3>Why Choose Style Sage?</h3>

        <ul>
            <li><strong>Convenience:</strong> Access expert fashion advice anytime, anywhere.</li>
            <li><strong>Personalization:</strong> Enjoy recommendations that are uniquely tailored to you.</li>
            <li><strong>Confidence:</strong> Step out with outfits that make you feel your best.</li>
            <li><strong>Sustainability:</strong> Make eco-friendly fashion choices with ease.</li>
            <li><strong>Innovation:</strong> Experience cutting-edge AI technology in the fashion world.</li>
        </ul>

        <h3>Join the Style Sage Community:</h3>

        <p>Embark on a journey to discover your best-dressed self with Style Sage. Connect with a community of fashion-forward individuals, share your looks, get inspired, and inspire others. Fashion is not just about clothing; it's about expressing who you are. Let Style Sage help you tell your story through impeccable style.</p>
    """, unsafe_allow_html=True)
    
    st.title("StyleSage👗 - Your Fashion Oracle")

    # Define paths
    base_path = 'C:\Projects\Fashion_recomendation\women_fashion\women fashion'
    csv_path = 'C:\Projects\Fashion_recomendation\women_fashion\women_fashion_image_names.csv'

    # Read the CSV file
    df = pd.read_csv(csv_path)

    # Create a dictionary to hold image paths
    image_dict = create_image_dict(base_path, df)

    # Load the LLM model
    llm = load_llm_model()

    # Fit a TF-IDF Vectorizer on the prompts
    vectorizer, prompt_vectors = load_vectorizer(df)

    # User interaction
    user_prompt = st.text_input("Revolutionize your style:", "party wear")
    num_images = st.slider("Set the stage with style: How many images would you like to see?", min_value=1, max_value=10, value=3)

    if st.button("Find Matches", key="find_matches_btn"):
        matched_rows = match_prompt(user_prompt, df, vectorizer, prompt_vectors, num_images)
        if matched_rows.empty:
            st.warning("No matches found. Please try a different prompt.")
        else:
            # Display matched images
            display_images(llm, image_dict, matched_rows)

    if st.button("Clear Cache", key="clear_cache_btn"):
        st.cache_resource.clear()
        st.cache_data.clear()
        st.success("Cache cleared! Please reload the app.")

# Run the app
if __name__ == "__main__":
    main()
