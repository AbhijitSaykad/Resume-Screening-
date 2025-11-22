import streamlit as st
import pickle
import docx
import PyPDF2
import re
import os

# ----------------------
# Paths for model files
# ----------------------
# Use the folder where app.py is located
model_dir = os.path.dirname(os.path.abspath(__file__))

clf_path = os.path.join(model_dir, 'clf.pkl')
tfidf_path = os.path.join(model_dir, 'tfidf.pkl')
encoder_path = os.path.join(model_dir, 'encoder.pkl')

# ----------------------
# Load models
# ----------------------
try:
    with open(clf_path, 'rb') as f:
        svc_model = pickle.load(f)
    with open(tfidf_path, 'rb') as f:
        tfidf = pickle.load(f)
    with open(encoder_path, 'rb') as f:
        le = pickle.load(f)
except FileNotFoundError:
    st.error(f"Model files not found in: {model_dir}")
    raise

# ----------------------
# Resume cleaning function
# ----------------------
def cleanResume(txt):
    cleanText = re.sub(r'http\S+\s', ' ', txt)
    cleanText = re.sub(r'RT|cc', ' ', cleanText)
    cleanText = re.sub(r'#\S+\s', ' ', cleanText)
    cleanText = re.sub(r'@\S+', ' ', cleanText)
    cleanText = re.sub(r'[%s]' % re.escape("""!"#$%&'()*+,-./:;<=>?@[\]^_`{|}~"""), ' ', cleanText)
    cleanText = re.sub(r'[^\x00-\x7f]', ' ', cleanText)
    cleanText = re.sub(r'\s+', ' ', cleanText)
    return cleanText.strip()

# ----------------------
# File extraction functions
# ----------------------
def extract_text_from_pdf(file):
    pdf_reader = PyPDF2.PdfReader(file)
    text = ''
    for page in pdf_reader.pages:
        page_text = page.extract_text()
        if page_text:
            text += page_text + ' '
    return text

def extract_text_from_docx(file):
    doc = docx.Document(file)
    text = ''
    for paragraph in doc.paragraphs:
        text += paragraph.text + '\n'
    return text

def extract_text_from_txt(file):
    try:
        text = file.read().decode('utf-8')
    except UnicodeDecodeError:
        text = file.read().decode('latin-1')
    return text

def handle_file_upload(uploaded_file):
    file_extension = uploaded_file.name.split('.')[-1].lower()
    if file_extension == 'pdf':
        return extract_text_from_pdf(uploaded_file)
    elif file_extension == 'docx':
        return extract_text_from_docx(uploaded_file)
    elif file_extension == 'txt':
        return extract_text_from_txt(uploaded_file)
    else:
        raise ValueError("Unsupported file type. Please upload a PDF, DOCX, or TXT file.")

# ----------------------
# Prediction function
# ----------------------
def pred(input_resume):
    cleaned_text = cleanResume(input_resume)
    vectorized_text = tfidf.transform([cleaned_text])
    vectorized_text = vectorized_text.toarray()
    predicted_category = svc_model.predict(vectorized_text)
    return le.inverse_transform(predicted_category)[0]

# ----------------------
# Streamlit app
# ----------------------
def main():
    st.set_page_config(page_title="📄 Resume Category Prediction", page_icon="🧠", layout="centered")

    # Title
    st.markdown("""
        <div style="text-align: center;">
            <h1 style="color: #264653;">📂 Resume Category Prediction</h1>
            <p style="color: gray;">Upload a resume to predict its category</p>
        </div>
        <hr style="border: 1px solid #e0e0e0;">
    """, unsafe_allow_html=True)

    # File uploader
    uploaded_file = st.file_uploader("📤 Upload Resume File (PDF, DOCX, or TXT)", type=["pdf", "docx", "txt"])

    if uploaded_file is not None:
        try:
            resume_text = handle_file_upload(uploaded_file)
            st.success("✅ Resume processed successfully!")

            category = pred(resume_text)

            # Display result
            st.markdown(f"""
                <div style="text-align: center; margin-top: 40px;">
                    <h3 style="color: #2a9d8f;">Predicted Category</h3>
                    <div style="font-size: 28px; font-weight: bold; color: #264653;
                                padding: 10px 20px; border: 2px solid #2a9d8f; border-radius: 8px;
                                display: inline-block;">
                        {category}</div>
                </div>
            """, unsafe_allow_html=True)

        except Exception as e:
            st.error(f"❌ Error processing the file: {str(e)}")
    else:
        st.info("👈 Upload a file to begin.")

if __name__ == "__main__":
    main()
