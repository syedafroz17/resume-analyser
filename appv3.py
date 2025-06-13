import streamlit as st
import PyPDF2
import docx2txt
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
import os

# Load environment variables from .env file
load_dotenv()

# Streamlit page configuration
st.set_page_config(
    page_title="Resume Analyzer",
    layout="wide",
    page_icon="📄"
)

# --- DATA EXTRACTION FUNCTIONS ---
def extract_text_from_pdf(file):
    try:
        pdf_reader = PyPDF2.PdfReader(file)
        text = "".join(page.extract_text() or "" for page in pdf_reader.pages)
        return text
    except Exception as e:
        st.error(f"Error reading PDF: {str(e)}")
        return None

def extract_text_from_docx(file):
    try:
        return docx2txt.process(file)
    except Exception as e:
        st.error(f"Error reading DOCX: {str(e)}")
        return None

# --- AI & PROMPT FUNCTIONS ---
def initialize_llm():
    try:
        api_key = os.getenv("GOOGLE_API_KEY")
        if not api_key:
            st.error("Google API key not found. Please set it in your environment variables.")
            return None
        return ChatGoogleGenerativeAI(model="gemini-1.5-pro", google_api_key=api_key,
                                    generation_config={"response_mime_type": "text/plain"})
    except Exception as e:
        st.error(f"Error initializing LLM: {str(e)}")
        return None

prompt_template = """
As an expert hiring manager, analyze the following resume against the provided job description. Provide a detailed, professional-grade response covering these points:

1.  **Initial Analysis**: Concisely summarize the candidate's profile and its relevance to the job description.
2.  **Hiring Manager's Reaction**:
    * **Positive Aspects**: What specific skills, experiences, or qualities make this candidate stand out?
    * **Areas for Concern**: What are the negative aspects or red flags that could hinder the candidate's chances?
3.  **Actionable Suggestions for Improvement**:
    * Provide concrete, specific advice on how the candidate can improve their resume to better align with this job role.
4.  **Overall Rating**: Rate the resume's suitability for the job on a scale of 1-10.

**Job Description**:
{job_description}

**Resume**:
{resume}

---
**YOUR ANALYSIS**:
"""

# --- MAIN APP LOGIC ---
def main():
    st.title("Sarah AI - Resume Analyzer")
    
    # File uploader for resume
    resume_file = st.file_uploader("Upload Resume (PDF or DOCX)", type=["pdf", "docx"])
    
    # Text area for job description
    job_description = st.text_area(
        "Job Description",
        height=200,
        placeholder="Paste the complete job description including requirements, responsibilities, and qualifications..."
    )
    
    # Analyze button
    if st.button("Analyze Resume", disabled=not (resume_file and job_description)):
        with st.spinner("Analyzing your resume..."):
            resume_text = ""
            if resume_file.type == "application/pdf":
                resume_text = extract_text_from_pdf(resume_file)
            else:
                resume_text = extract_text_from_docx(resume_file)

            if resume_text and job_description:
                llm = initialize_llm()
                if llm:
                    try:
                        prompt = PromptTemplate(
                            input_variables=["job_description", "resume"],
                            template=prompt_template
                        )
                        chain = LLMChain(llm=llm, prompt=prompt)
                        result = chain.run(job_description=job_description, resume=resume_text)
                        st.subheader("Analysis Results")
                        st.write(result)
                    except Exception as e:
                        st.error(f"Error during analysis: {str(e)}")
                else:
                    st.error("Failed to initialize AI model. Please check your API key.")
            else:
                st.error("Failed to extract text from documents. Please try again.")

if __name__ == "__main__":
    main()
