import streamlit as st
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
import numpy as np
from PIL import Image, ImageOps
import matplotlib.pyplot as plt
import cv2
from gtts import gTTS
import os
import base64
import time  # Import time module to fix the error
import google.generativeai as genai

# Define disease options with alternating colors
disease_options = {
    "Skin Cancer Detection": ("yellow", "Skin Cancer Detection"),
    "Brain Tumor Detection": ("orange", "Brain Tumor Detection"),
    "Lung Disease Detection (Pneumonia)": ("yellow", "Lung Disease Detection (Pneumonia)"),
    "Eye Disease Detection": ("yellow", "Eye Disease Detection")
}
st.title('🩺 Multiple Disease Detection')
# Sidebar for navigation
selected_option = st.selectbox(
    'Select Disease to Detect:',
    ['Skin Cancer Detection', 'Brain Tumor Detection', 'Lung Disease Detection (Pneumonia)', 'Eye Disease Detection']
)


# Load the trained models
skin_model = load_model("skin_disease_cnn_model.keras")
brain_model = load_model("brain_tumor_cnn_model.keras")
lung_model = load_model("lung_pneumonia.keras")
eye_model = load_model("eye_disease_model.keras")

# Define categories
skin_categories = ['Actinic Keratosis', 'Basal Cell Carcinoma', 'Dermatofibroma', 'Melanoma', 'Nevus',
                   'Pigmented Benign Keratosis', 'Seborrheic Keratosis', 'Squamous Cell Carcinoma', 'Vascular Lesion']
brain_categories = ['Glioma', 'Meningioma', 'Pituitary', 'No Tumor']
lung_categories = ['Pneumonia', 'Normal']
eye_categories = ["Cataract", "Diabetic Retinopathy", "Glaucoma", "Normal"]


def show_health_advice(predicted_class, category):
    health_tips = {
    "Skin Diseases": {
        'Actinic Keratosis': [
            "Wear sunscreen daily (SPF 30+).",
            "Avoid prolonged sun exposure, especially midday.",
            "Visit a dermatologist for early treatment."
        ],
        'Basal Cell Carcinoma': [
            "Early detection is key—schedule regular skin check-ups.",
            "Protect your skin with UV-blocking clothing and sunscreen.",
            "Seek medical treatment for removal options."
        ],
        'Dermatofibroma': [
            "Generally harmless but monitor for changes in size or color.",
            "Avoid scratching or irritating the area.",
            "Consult a dermatologist if it becomes painful."
        ],
        'Melanoma': [
            "Check moles for asymmetry, border irregularity, and color changes.",
            "Use SPF 50+ sunscreen and wear protective clothing.",
            "Consult a doctor immediately for suspicious skin changes."
        ],
        'Nevus': [
            "Monitor for changes in size, shape, or color.",
            "Avoid excessive sun exposure to prevent complications.",
            "See a dermatologist if the nevus becomes itchy or grows."
        ],
        'Pigmented Benign Keratosis': [
            "Usually harmless, but consult a doctor for unusual growths.",
            "Use moisturizers to keep the skin healthy.",
            "Consider removal for cosmetic reasons if needed."
        ],
        'Seborrheic Keratosis': [
            "Harmless but can be removed for cosmetic reasons.",
            "Avoid scratching or picking at the lesion.",
            "Moisturize the skin to reduce irritation."
        ],
        'Squamous Cell Carcinoma': [
            "Protect your skin from UV rays with sunscreen and clothing.",
            "Regularly check for new or changing skin lesions.",
            "Seek prompt medical treatment if diagnosed."
        ],
        'Vascular Lesion': [
            "Laser treatments can help improve appearance.",
            "Keep the skin hydrated to reduce irritation.",
            "Consult a dermatologist for treatment options."
        ]
    },
    "Brain Diseases": {
        'Glioma': [
            "Maintain a healthy diet rich in antioxidants.",
            "Exercise regularly to improve brain health.",
            "Schedule regular neurological check-ups."
        ],
        'Meningioma': [
            "Monitor symptoms such as headaches or vision changes.",
            "Discuss surgical and non-surgical treatment options with your doctor.",
            "Adopt a brain-healthy lifestyle with a balanced diet and exercise."
        ],
        'Pituitary': [
            "Regular hormone check-ups help manage symptoms.",
            "Maintain a balanced diet to support endocrine health.",
            "Follow prescribed medications or treatments."
        ],
        'No Tumor': [
            "Great news! Maintain a healthy diet with brain-boosting foods.",
            "Stay active and engage in mental exercises.",
            "Get regular check-ups to ensure long-term health."
        ]
    },
    "Lung Diseases": {
        'Pneumonia': [
            "Rest and stay hydrated to aid recovery.",
            "Follow prescribed antibiotics and medications.",
            "Avoid smoking and exposure to pollutants."
        ],
        'Normal': [
            "Lungs are healthy! Avoid smoking and air pollution.",
            "Practice deep breathing exercises for lung health.",
            "Maintain an active lifestyle to keep lungs strong."
        ]
    },
    "Eye Diseases": {
        'Cataract': [
            "Eat vitamin A-rich foods like carrots and leafy greens.",
            "Wear UV-protected sunglasses to prevent worsening.",
            "Schedule regular eye check-ups for early detection."
        ],
        'Diabetic Retinopathy': [
            "Manage blood sugar levels through a healthy diet.",
            "Get regular eye check-ups to monitor vision changes.",
            "Exercise regularly to improve circulation."
        ],
        'Glaucoma': [
            "Use prescribed eye drops consistently.",
            "Take breaks from screens to reduce eye strain.",
            "Eat omega-3-rich foods to support eye health."
        ],
        'Normal': [
            "Your eyes are healthy! Reduce screen time to prevent strain.",
            "Follow the 20-20-20 rule (every 20 minutes, look 20 feet away for 20 seconds).",
            "Get regular eye exams to maintain vision health."
        ]
    }
}
    st.markdown(f"""
        <div style="background-color:#F8F9FA; padding:15px; border-radius:10px; box-shadow: 2px 2px 10px #888;">
            <h3 style="color:#2C3E50;">📝 Health Advice for {predicted_class}</h3>
            <p style="font-size:16px; color:#555;">{health_tips[category].get(predicted_class, "Stay healthy and consult a doctor if needed.")}</p>
            <blockquote style="font-style:italic; color:#777;">"Take care of your body. It’s the only place you have to live." </blockquote>
        </div>
    """, unsafe_allow_html=True)



# Function to plot prediction probabilities
def plot_prediction(predictions, categories, title):
    categories = list(categories)  # Ensure categories is a list
    plt.figure(figsize=(8, 5))
    plt.bar(categories, predictions[0] * 100, color=['#FF6347', '#4682B4', '#32CD32', '#FFD700'])
    plt.xticks(rotation=45, ha='right')
    plt.title(title)
    plt.xlabel('Disease Type')
    plt.ylabel('Confidence (%)')
    st.pyplot(plt)



# Replace with your actual API key
genai.configure(api_key="AIzaSyDnMsQoYi5N0weH_uFpDazUc6B039YFC1A")

def chat_with_gemini(prompt, chat_history):
    model = genai.GenerativeModel("gemini-2.0-flash")  # Using Pro model for better responses
    response = model.generate_content(prompt)
    chat_history.append(("You", prompt))
    chat_history.append(("health bot", response.text))
    return response.text

# Sidebar Chatbot
st.sidebar.title("🤖 Health Chatbot")
st.sidebar.subheader("Ask me anything about Queries!")
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# Display chat history inside the sidebar
with st.sidebar:
    chat_container = st.container()
    with chat_container:
        for role, message in st.session_state.chat_history:
            with st.chat_message(role):
                st.write(message)

    # Stylish input box within the sidebar
    user_input = st.text_input(
        "", "",
        key="user_input",
        placeholder="Ask anything...",
        help="Type your message and press Enter"
    )
    send_button = st.button("🎤", help="Getting as soon as possible")

    if user_input:
        response = chat_with_gemini(user_input, st.session_state.chat_history)
        # Keep only the latest Q&A pair
        st.session_state.chat_history = [("You", user_input), ("health bot", response)]
        with chat_container:
            with st.chat_message("health bot"):
                st.write(response)

        

# User feedback
st.sidebar.markdown("### 🤔 Was this helpful?")
feedback = st.sidebar.radio("", ["Yes", "No"])
if feedback == "Yes":
    st.sidebar.success("Glad I could help! 💛")
elif feedback == "No":
    st.sidebar.warning("I'll keep improving! 🚀")

st.sidebar.markdown('---')


# Define disease categories and their symptoms
disease_symptoms = {
    "Skin Diseases": {
        'Actinic Keratosis': "Rough, scaly patches on skin, often caused by sun exposure.",
        'Basal Cell Carcinoma': "Shiny or waxy bumps, open sores that don’t heal.",
        'Dermatofibroma': "Firm, reddish-brown nodules on the skin.",
        'Melanoma': "Irregularly shaped or multicolored moles, changes in existing moles.",
        'Nevus': "Small, pigmented skin growths, usually harmless.",
        'Pigmented Benign Keratosis': "Dark, warty, or scaly spots on the skin.",
        'Seborrheic Keratosis': "Noncancerous, scaly, and pigmented growths.",
        'Squamous Cell Carcinoma': "Firm red nodules, scaly patches that bleed easily.",
        'Vascular Lesion': "Abnormal blood vessel growths, often red or purple in color."
    },
    "Brain Diseases": {
        'Glioma': "Headaches, nausea, memory issues, personality changes.",
        'Meningioma': "Seizures, vision problems, headaches, memory loss.",
        'Pituitary': "Hormonal imbalances, vision changes, headaches.",
        'No Tumor': "No symptoms present."
    },
    "Lung Diseases": {
        'Pneumonia': "Cough with phlegm, fever, chills, difficulty breathing.",
        'Normal': "No symptoms present."
    },
    "Eye Diseases": {
        'Cataract': "Cloudy or blurred vision, difficulty seeing at night.",
        'Diabetic Retinopathy': "Blurred vision, dark spots, fluctuating vision.",
        'Glaucoma': "Loss of peripheral vision, eye pain, headache.",
        'Normal': "No symptoms present."
    }
}

# Sidebar Symptom Checker
st.sidebar.subheader("💬 Symptom Checker")
category = st.sidebar.selectbox("Select Disease Category", list(disease_symptoms.keys())) # Select category
# Select disease within category
disease = st.sidebar.selectbox("Select Disease", list(disease_symptoms[category].keys()))
# Display symptoms
#st.sidebar.text_area.write(disease_symptoms[category][disease])("Default Symptoms are:")
# Pre-fill symptoms in text area
default_text = f"Symptoms: {disease_symptoms[category][disease]}"

st.sidebar.text_area("Default Symptoms:", default_text, height=150)

st.sidebar.markdown('---')

st.sidebar.markdown('*About the App*')
st.sidebar.info("This app uses a Convolutional Neural Network (CNN) model to classify diseases into Brain Tumor, Pneumonia, Eye Disease and Skin Cancer")


# Streamlit UI
st.write('Choose a disease detection method below to upload an image for classification.')

# Animation for hearts effect
HEART_ANIMATION = """
<style>
@keyframes hearts {
    0% { transform: translateY(0) scale(1); opacity: 1; }
    100% { transform: translateY(-100vh) scale(2); opacity: 0; }
}
.heart {
    position: fixed;
    bottom: 0;
    left: 50%;
    font-size: 2rem;
    color: red;
    animation: hearts 3s linear infinite;
}
</style>
<div class='heart'>❤️</div>
<div class='heart' style='left: 30%; animation-delay: 0.5s;'>❤️</div>
<div class='heart' style='left: 70%; animation-delay: 1s;'>❤️</div>
"""

def show_hearts():
    st.markdown(HEART_ANIMATION, unsafe_allow_html=True)
    time.sleep(3)  # Display animation for 3 seconds

# Function for voice output using gTTS
# Function for voice output using gTTS
def speak(text):
    tts = gTTS(text=text, lang="en")
    tts.save("prediction.mp3")
    
    # Convert to base64 for automatic playback
    with open("prediction.mp3", "rb") as audio_file:
        audio_bytes = audio_file.read()
        encoded_audio = base64.b64encode(audio_bytes).decode()
    
    # Embed audio in an auto-playing HTML tag
    autoplay_audio = f"""
        <audio autoplay>
            <source src="data:audio/mp3;base64,{encoded_audio}" type="audio/mp3">
        </audio>
    """
    st.markdown(autoplay_audio, unsafe_allow_html=True)

# Preprocessing functions
def preprocess_skin_image(img):
    img = img.resize((224, 224))
    img_array = img_to_array(img) / 255.0
    return np.expand_dims(img_array, axis=0)

def preprocess_brain_image(img):
    img = img.convert("RGB").resize((224, 224))
    img_array = np.array(img) / 255.0
    return np.expand_dims(img_array, axis=0)

def preprocess_kidney_image(img):
    img = img.convert("RGB")
    img = np.array(img)
    img = cv2.resize(img, (28, 28)) / 255.0
    return np.expand_dims(img, axis=0)

def preprocess_lung_image(img):
    img = img.convert("RGB")
    img = np.array(img)
    img = cv2.resize(img, (300, 300)) / 255.0
    return np.expand_dims(img, axis=0)
    
def preprocess_eye_image(img):
    img = img.convert("RGB")
    img = np.array(img)
    img = cv2.resize(img, (224, 224)) / 255.0
    return np.expand_dims(img, axis=0) 

# Function for processing images
def process_and_predict(uploaded_file, model, preprocess_func, categories, title, category_name):
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Image", use_container_width=True)
        st.write("Classifying... Please wait while the model processes the image.")

        # Preprocess image
        img_array = preprocess_func(image)

        # Make prediction
        predictions = model.predict(img_array)
        predicted_class = categories[np.argmax(predictions)]
        confidence = np.max(predictions) * 100

        # Display result
        st.subheader(f"🔮 Prediction: **{predicted_class}** ")

        # Plot prediction probabilities
        plot_prediction(predictions, categories, title)
        

        show_health_advice(predicted_class, category_name)
        
        # Voice output
        speak(f"The predicted disease is {predicted_class}.")

        # Feedback buttons
        st.markdown("### 📝 Hoping it is Helpful? ")
        col1, col2 = st.columns(2)
        with col1:
            if st.button("✅ Yes", key=f"yes_{predicted_class}"):
                show_hearts()
                st.success("Glad I Could Help!😊")
        
        with col2:
            if st.button("❌ No", key=f"no_{predicted_class}"):
                st.warning("We will improve our model. Thank you for your feedback! 🙏")

# Disease Detection Sections
if selected_option == 'Skin Cancer Detection':
    st.markdown('<h2 style="color:#FAD02E;"> 🔬 Skin Cancer Detection</h2>', unsafe_allow_html=True)
    uploaded_file = st.file_uploader("Upload a skin image...", type=['jpg', 'png', 'jpeg'])
    process_and_predict(uploaded_file, skin_model, preprocess_skin_image, skin_categories,"Skin Cancer Prediction Probabilities", "Skin Diseases")    

elif selected_option == 'Brain Tumor Detection':
    st.markdown('<h2 style="color:#FF5C5C;"> 🧠 Brain Tumor Detection</h2>', unsafe_allow_html=True)
    uploaded_file = st.file_uploader("Upload an MRI scan...", type=['jpg', 'png', 'jpeg'])
    process_and_predict(uploaded_file, brain_model, preprocess_brain_image, brain_categories,"Brain Tumor Prediction Probabilities", "Brain Diseases")

elif selected_option == 'Lung Disease Detection (Pneumonia)':
    st.markdown('<h2 style="color:#FFDD00;"> 🫁 Lung Disease Detection (Pneumonia)</h2>', unsafe_allow_html=True)
    uploaded_file = st.file_uploader("Upload a Chest X-ray...", type=["jpg", "png", "jpeg"])
    process_and_predict(uploaded_file, lung_model, preprocess_lung_image, lung_categories,"Lung Disease Prediction Probabilities", "Lung Diseases")
    
elif selected_option == 'Eye Disease Detection':
    st.markdown('<h2 style="color:#FF5733;"> 🔍 Eye Disease Detection</h2>', unsafe_allow_html=True)
    uploaded_file = st.file_uploader("Upload an Eye Image...", type=["jpg", "png", "jpeg"])
    process_and_predict(uploaded_file, eye_model, preprocess_eye_image, eye_categories,"Eye Disease Prediction Probabilities", "Eye Diseases")

# Footer
st.markdown('<br><hr><h4 style="color:#999;">Made with ❤️ by You</h4>', unsafe_allow_html=True)
