import streamlit as st
import PyPDF2
import docx2txt
from langchain_core.prompts import PromptTemplate
from langchain_community.llms import OpenAI
from langchain_anthropic import ChatAnthropic
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.runnables import RunnableSequence
import io
from datetime import datetime

# Streamlit page configuration
st.set_page_config(page_title="Resume Analyzer", layout="wide")

# Initialize session state
if 'llm_provider' not in st.session_state:
    st.session_state.llm_provider = None
if 'model_name' not in st.session_state:
    st.session_state.model_name = None
if 'api_key' not in st.session_state:
    st.session_state.api_key = None
if 'analysis_result' not in st.session_state:
    st.session_state.analysis_result = None
if 'resume_text' not in st.session_state:
    st.session_state.resume_text = None
if 'job_description' not in st.session_state:
    st.session_state.job_description = None

# Available LLM providers and their models
llm_providers = {
    "OpenAI": ["gpt-3.5-turbo", "gpt-4", "gpt-4-turbo"],
    "Grok": ["grok-3"],  # Placeholder for Grok
    "Anthropic": ["claude-3-opus", "claude-3-sonnet", "claude-3-haiku"],
    "Gemini": ["gemini-1.5-pro", "gemini-1.5-flash"]
}

# LaTeX resume template (provided by user)
latex_template = r"""
\documentclass[a4paper,10pt]{article}

\usepackage{latexsym}
\usepackage[empty]{fullpage}
\usepackage{titlesec}
\usepackage{marvosym}
\usepackage[usenames,dvipsnames]{color}
\usepackage{verbatim}
\usepackage{enumitem}
\usepackage[hidelinks]{hyperref}
\usepackage{fancyhdr}
\usepackage[english]{babel}
\usepackage{tabularx}
\usepackage{fontawesome5}
\usepackage{multicol}
\setlength{\multicolsep}{-3.0pt}
\setlength{\columnsep}{-1pt}
\input{glyphtounicode}

%----------FONT OPTIONS----------
\usepackage{times}

\pagestyle{fancy}
\fancyhf{}
\fancyfoot{}
\renewcommand{\headrulewidth}{0pt}
\renewcommand{\footrulewidth}{0pt}

% Adjust margins
\addtolength{\oddsidemargin}{-0.6in}
\addtolength{\evensidemargin}{-0.5in}
\addtolength{\textwidth}{1.19in}
\addtolength{\topmargin}{-.8in}
\addtolength{\textheight}{1.5in}

\urlstyle{same}

\raggedbottom
\raggedright
\setlength{\tabcolsep}{0in}

% Section formatting
\titleformat{\section}{
  \vspace{-4pt}\scshape\raggedright\large\bfseries
}{}{0em}{}[\color{black}\titlerule \vspace{-5pt}]

% Ensure PDF is ATS parsable
\pdfgentounicode=1

%-------------------------
% Custom commands
\newcommand{\resumeItem}[1]{
  \item\small{{#1 \vspace{-2pt}}}
}
\newcommand{\resumeSubheading}[4]{
  \vspace{-2pt}\item
    \begin{tabular*}{1.0\textwidth}[t]{l@{\extracolsep{\fill}}r}
      \textbf{#1} & \textbf{\small #2} \\
      \textit{\small#3} & \textit{\small #4} \\
    \end{tabular*}\vspace{-7pt}
}
\newcommand{\resumeProjectHeading}[2]{
    \item
    \begin{tabular*}{1.0\textwidth}{l@{\extracolsep{\fill}}r}
      \small#1 & \textbf{\small #2} \\
    \end{tabular*}\vspace{-7pt}
}
\newcommand{\resumeSubItem}[1]{\resumeItem{#1}\vspace{-4pt}}
\renewcommand\labelitemi{$\vcenter{\hbox{\tiny$\bullet$}}$}
\renewcommand\labelitemii{$\vcenter{\hbox{\tiny$\bullet$}}$}

\newcommand{\resumeSubHeadingListStart}{\begin{itemize}[leftmargin=0.0in, label={}]}
\newcommand{\resumeSubHeadingListEnd}{\end{itemize}}
\newcommand{\resumeItemListStart}{\begin{itemize}}
\newcommand{\resumeItemListEnd}{\end{itemize}\vspace{-5pt}}

%-------------------------------------------
%%%%%%  RESUME STARTS HERE  %%%%%%%%%%%%%%%%%%%%%%%%%%%%

\begin{document}

%----------HEADER----------
\begin{center}
    {\Huge \scshape {name}} \\ \vspace{1pt}
    \small \raisebox{-0.1\height}\faPhone\ {phone} ~
    \href{mailto:{email}}{\raisebox{-0.2\height}\faEnvelope\ {email}} ~
    \href{https://linkedin.com/in/{linkedin}}{\raisebox{-0.2\height}\faLinkedin\ LinkedIn} ~
    \href{https://github.com/{github}}{\raisebox{-0.2\height}\faGithub\ GitHub} ~
    \small {location} \\
    \vspace{-5pt}
\end{center}

%-----------PROFESSIONAL SUMMARY-----------
\section{Professional Summary}
\resumeItemListStart
    \resumeItem{{summary}}
\resumeItemListEnd

%-----------EXPERIENCE-----------
\section{Experience}
\resumeSubHeadingListStart
{experience}
\resumeSubHeadingListEnd

%-----------PROJECTS-----------
\section{Projects}
\resumeSubHeadingListStart
{projects}
\resumeSubHeadingListEnd

%-----------SKILLS-----------
\section{Skills}
\resumeItemListStart
{skills}
\resumeItemListEnd

%-----------EDUCATION-----------
\section{Education}
\resumeSubHeadingListStart
{education}
\resumeSubHeadingListEnd

\end{document}
"""

# Function to extract text from PDF
def extract_text_from_pdf(file):
    try:
        pdf_reader = PyPDF2.PdfReader(file)
        text = ""
        for page in pdf_reader.pages:
            text += page.extract_text() or ""
        return text
    except Exception as e:
        st.error(f"Error reading PDF: {str(e)}")
        return None

# Function to extract text from DOCX
def extract_text_from_docx(file):
    try:
        text = docx2txt.process(file)
        return text
    except Exception as e:
        st.error(f"Error reading DOCX: {str(e)}")
        return None

# Function to initialize LLM
def initialize_llm(provider, model_name, api_key):
    try:
        if provider == "OpenAI":
            return OpenAI(model=model_name, api_key=api_key)
        elif provider == "Grok":
            st.warning("Grok API is a placeholder. Please replace with actual xAI API.")
            return None
        elif provider == "Anthropic":
            return ChatAnthropic(model=model_name, api_key=api_key)
        elif provider == "Gemini":
            return ChatGoogleGenerativeAI(model=model_name, google_api_key=api_key)
        else:
            st.error("Invalid LLM provider selected.")
            return None
    except Exception as e:
        st.error(f"Error initializing LLM: {str(e)}")
        return None

# Prompt template for resume analysis
analysis_prompt_template = """
As a hiring manager, analyze the following resume against the provided job description. Provide a detailed response covering:

1. **Initial Analysis**: Review the job description and resume thoroughly.
2. **Reaction as a Hiring Manager**:
   - Describe your reaction to the resume.
   - Identify positive aspects that increase the candidate's chances of getting hired.
   - Point out negative aspects that could hinder the candidate's chances.
3. **Suggestions for Improvement**:
   - Provide specific improvements to increase the candidate's chances of getting hired.
   - Offer suggestions to enhance the resume.
4. **Rating**: Rate the resume on a scale of 1-10 based on its alignment with the job description and overall quality.

**Job Description**:
{job_description}

**Resume**:
{resume}

**Output Format**:
- **Initial Analysis**: [Your analysis]
- **Reaction as a Hiring Manager**:
  - **Positive Aspects**: [List positive aspects]
  - **Negative Aspects**: [List negative aspects]
- **Suggestions for Improvement**: [List suggestions]
- **Rating**: [Rating out of 10]
"""

# Prompt template for rewriting resume
rewrite_prompt_template = """
You are tasked with rewriting a resume to align with a given job description, using the provided LaTeX template, while ensuring the content remains ATS-friendly. Only modify existing information by rephrasing descriptions of internships, work experience, and projects to better match the job description, and add relevant keywords to the skills section. Do not add new experiences or information not present in the original resume. Incorporate the user's specific requests for emphasis or changes and the analysis suggestions.

**Job Description**:
{job_description}

**Original Resume**:
{resume}

**Analysis Suggestions**:
{suggestions}

**User Requests**:
{user_requests}

**LaTeX Template**:
{latex_template}

**Instructions**:
- Rewrite the resume to fit the LaTeX template structure, using commands like \resumeItem, \resumeSubheading, and \resumeProjectHeading.
- For each experience and project, use \resumeSubheading{Title}{Dates}{Role/Org}{Location}\resumeItemListStart\resumeItem{Description}...\resumeItemListEnd.
- For skills, use \resumeItem{Category: Skill list}.
- For education, use \resumeSubheading{University}{Dates}{Degree}{Location}.
- Modify descriptions of internships, work experience, and projects to align with the job description.
- Add relevant keywords to the skills section that match the job description.
- Ensure all content is based on the original resume; no new experiences or information should be added.
- Return only the populated LaTeX code between \begin{document} and \end{document}, with all sections formatted correctly.

**Output**:
LaTeX code for the resume, starting from \begin{document} and ending with \end{document}.
"""

# Streamlit UI - Sidebar
st.sidebar.header("LLM Configuration")
llm_provider = st.sidebar.selectbox(
    "Select LLM Provider",
    options=["Select a provider"] + list(llm_providers.keys()),
    index=0
)

# Update session state for LLM provider
if llm_provider != "Select a provider":
    st.session_state.llm_provider = llm_provider
else:
    st.session_state.llm_provider = None

# Model selection
if st.session_state.llm_provider:
    model_name = st.sidebar.selectbox(
        "Select Model",
        options=llm_providers[st.session_state.llm_provider],
        index=0
    )
    st.session_state.model_name = model_name

    # API key input
    api_key = st.sidebar.text_input(
        f"Enter {st.session_state.llm_provider} API Key",
        type="password"
    )
    st.session_state.api_key = api_key if api_key else None
else:
    st.session_state.model_name = None
    st.session_state.api_key = None

# Main content
st.title("Resume Analyzer AI Agent")
st.header("Upload Documents")

# Resume upload
resume_file = st.file_uploader("Upload Resume (PDF or DOCX)", type=["pdf", "docx"])

# Job description input
jd_option = st.radio("Job Description Input", ["Paste Text", "Upload File"])
job_description = ""
if jd_option == "Paste Text":
    job_description = st.text_area("Paste Job Description", height=200)
else:
    jd_file = st.file_uploader("Upload Job Description (PDF or Text)", type=["pdf", "txt"])
    if jd_file:
        if jd_file.type == "application/pdf":
            job_description = extract_text_from_pdf(jd_file)
        else:
            job_description = jd_file.read().decode("utf-8")

# Analyze button
if st.button("Analyze Resume", disabled=not (st.session_state.llm_provider and st.session_state.model_name and st.session_state.api_key and resume_file and job_description)):
    with st.spinner("Analyzing..."):
        # Extract resume text
        resume = ""
        if resume_file:
            if resume_file.type == "application/pdf":
                resume = extract_text_from_pdf(resume_file)
            else:
                resume = extract_text_from_docx(resume_file)

        if resume and job_description:
            # Store resume and JD in session state
            st.session_state.resume_text = resume
            st.session_state.job_description = job_description

            # Initialize LLM
            llm = initialize_llm(st.session_state.llm_provider, st.session_state.model_name, st.session_state.api_key)
            if llm:
                # Create analysis chain
                analysis_prompt = PromptTemplate(
                    input_variables=["job_description", "resume"],
                    template=analysis_prompt_template
                )
                analysis_chain = analysis_prompt | llm

                # Run analysis
                try:
                    result = analysis_chain.invoke({"job_description": job_description, "resume": resume})
                    st.session_state.analysis_result = result
                    st.subheader("Analysis Result")
                    st.write(result)
                except Exception as e:
                    st.error(f"Error during analysis: {str(e)}")
            else:
                st.error("Failed to initialize LLM. Please check API key and try again.")
        else:
            st.error("Failed to extract text from resume or job description.")

# Rewrite resume section
if st.session_state.analysis_result:
    st.subheader("Rewrite Resume")
    with st.form(key="rewrite_form"):
        st.write("Specify any experiences or sections to emphasize or modify for better alignment with the job description:")
        user_requests = st.text_area(
            "Preferences (e.g., emphasize specific internship/project, modify skills section, etc.)",
            placeholder="Example: Emphasize my data analysis internship at Company X, add machine learning keywords to skills."
        )
        submit_rewrite = st.form_submit_button("Rewrite Resume")

    if submit_rewrite:
        with st.spinner("Rewriting Resume..."):
            # Initialize LLM
            llm = initialize_llm(st.session_state.llm_provider, st.session_state.model_name, st.session_state.api_key)
            if llm:
                # Extract suggestions from analysis
                analysis_lines = st.session_state.analysis_result.split("\n")
                suggestions = ""
                capture_suggestions = False
                for line in analysis_lines:
                    if "Suggestions for Improvement" in line:
                        capture_suggestions = True
                        continue
                    if capture_suggestions and line.strip().startswith("- **"):
                        break
                    if capture_suggestions:
                        suggestions += line + "\n"

                # Create rewrite chain
                rewrite_prompt = PromptTemplate(
                    input_variables=["job_description", "resume", "suggestions", "user_requests", "latex_template"],
                    template=rewrite_prompt_template
                )
                rewrite_chain = rewrite_prompt | llm

                # Run rewrite
                try:
                    rewritten_resume = rewrite_chain.invoke({
                        "job_description": st.session_state.job_description,
                        "resume": st.session_state.resume_text,
                        "suggestions": suggestions,
                        "user_requests": user_requests,
                        "latex_template": latex_template
                    })

                    # Extract LaTeX content between \begin{document} and \end{document}
                    start_marker = r"\begin{document}"
                    end_marker = r"\end{document}"
                    start_idx = rewritten_resume.find(start_marker) + len(start_marker)
                    end_idx = rewritten_resume.find(end_marker)
                    if start_idx == -1 or end_idx == -1:
                        st.error("Failed to parse LaTeX output from LLM.")
                    else:
                        latex_resume = rewritten_resume[start_idx:end_idx].strip()

                        # Display LaTeX code
                        st.subheader("Rewritten Resume (LaTeX)")
                        st.code(latex_resume, language="latex")

                        # Provide download button for full LaTeX document
                        full_latex_resume = latex_template.replace(
                            r"\begin{document}" + "\n" + r"%----------HEADER----------" + "\n" + latex_resume + "\n" + r"\end{document}",
                            latex_resume
                        )
                        st.download_button(
                            label="Download LaTeX Resume",
                            data=full_latex_resume,
                            file_name=f"rewritten_resume_{datetime.now().strftime('%Y%m%d_%H%M%S')}.tex",
                            mime="text/plain"
                        )
                except Exception as e:
                    st.error(f"Error during resume rewriting: {str(e)}")
            else:
                st.error("Failed to initialize LLM for rewriting. Please check API key and try again.")

