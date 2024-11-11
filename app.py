import os
import streamlit as st
from llama_index.core import Settings, VectorStoreIndex, StorageContext, Document
from llama_index.core.node_parser import SentenceSplitter
from llama_index.vector_stores.milvus import MilvusVectorStore
from llama_index.embeddings.nvidia import NVIDIAEmbedding
from llama_index.llms.nvidia import NVIDIA
from llama_index.core.prompts import Prompt
from unstructured.partition.auto import partition
from unstructured.documents.elements import Text, Title
from pathlib import Path
import tempfile
from typing import List
import re
from datetime import datetime
from fpdf import FPDF
import textwrap

# Enhanced resume extraction prompt
RESUME_EXTRACTION_PROMPT = """
From the resume in the vector store, extract and organize the following information:

1. Education:
   - Degree(s)
   - University/Institution
   - Graduation year
   - GPA (if available)

2. Work Experience:
   - Company names
   - Job titles
   - Duration
   - Key responsibilities and achievements

3. Projects:
   - Project names
   - Technologies used
   - Key outcomes

4. Technical Skills:
   - Programming languages
   - Tools and frameworks
   - Other technical competencies

5. Certifications (if any)

Resume Context: {text}
Analyze and provide all available details from these categories.
"""

# Enhanced cover letter generation prompt
COVER_LETTER_PROMPT = """
Based on the candidate's resume details below and the provided job description, create a compelling cover letter. 
Follow this structure:

1. Opening: Brief introduction and position interest
2. Body Paragraph 1: Key relevant skills matching job requirements, highlighting educational background
3. Body Paragraph 2: Relevant work experiences and achievements that align with the role
4. Body Paragraph 3: Highlight relevant projects and technical expertise
5. Body Paragraph 4: Why this company specifically and how you can contribute
6. Closing: Call to action and thank you

Detailed Resume Information:
{resume_context}

Job Description:
{job_description}

Generate a professional cover letter that specifically highlights the candidate's qualifications matching the job requirements.
Focus on drawing clear connections between the candidate's experience and the job requirements.
"""

class PDF(FPDF):
    def header(self):
        pass
    
    def footer(self):
        pass

def extract_text_with_unstructured(file_path: str) -> str:
    """Extract text from documents using unstructured library"""
    elements = partition(filename=file_path)
    text_chunks = []
    
    for element in elements:
        if isinstance(element, (Text, Title)):
            text_chunks.append(str(element))
    
    return "\n".join(text_chunks)

def preprocess_text(text: str) -> str:
    """Clean and structure the extracted text"""
    # Remove multiple newlines and spaces
    text = re.sub(r'\n\s*\n', '\n\n', text)
    text = re.sub(r'\s+', ' ', text)
    
    # Identify and structure sections
    sections = {
        'education': r'(?i)education|academic|degree',
        'experience': r'(?i)experience|employment|work history',
        'skills': r'(?i)skills|technical skills|competencies',
        'projects': r'(?i)projects|portfolio',
        'certifications': r'(?i)certifications|certificates'
    }
    
    structured_text = ""
    for section, pattern in sections.items():
        matches = re.finditer(pattern, text, re.IGNORECASE)
        for match in matches:
            start = match.start()
            next_section_start = float('inf')
            
            # Find the start of the next section
            for other_pattern in sections.values():
                other_matches = re.finditer(other_pattern, text[start + len(match.group()):], re.IGNORECASE)
                for other_match in other_matches:
                    next_section_start = min(next_section_start, 
                                          start + len(match.group()) + other_match.start())
            
            section_text = text[start:next_section_start if next_section_start != float('inf') else None]
            structured_text += f"\n\n{section.upper()}:\n{section_text}"
    
    return structured_text

class EnhancedCoverLetterGenerator:
    def __init__(self):
        self.setup_settings()
        self.setup_vector_store()
        
    def setup_settings(self):
        """Initialize LlamaIndex settings with improved chunking"""
        Settings.embed_model = NVIDIAEmbedding(
            model="nvidia/nv-embedqa-e5-v5", 
            truncate="END"
        )
        Settings.llm = NVIDIA(
            model="meta/llama-3.1-70b-instruct",
            stream=True  # Enable streaming
        )
        Settings.text_splitter = SentenceSplitter(
            chunk_size=1500,
            chunk_overlap=200
        )
    
    def setup_vector_store(self):
        """Initialize Milvus vector store"""
        try:
            self.vector_store = MilvusVectorStore(
                host="localhost",
                port=19530,
                dim=1024,
                collection_name="resume_vectors"
            )
            self.storage_context = StorageContext.from_defaults(
                vector_store=self.vector_store
            )
        except Exception as e:
            st.error(f"Failed to setup vector store: {str(e)}")
            self.storage_context = StorageContext.from_defaults()
    
    def create_documents(self, text: str, metadata: dict = None) -> List[Document]:
        """Create LlamaIndex Document objects from text"""
        if metadata is None:
            metadata = {}
        return [Document(text=text, metadata=metadata)]
    
    def process_uploaded_file(self, uploaded_file) -> str:
        """Process uploaded file and extract text using unstructured"""
        with tempfile.NamedTemporaryFile(delete=False, suffix=Path(uploaded_file.name).suffix) as tmp_file:
            tmp_file.write(uploaded_file.getvalue())
            tmp_file_path = tmp_file.name
            
        try:
            extracted_text = extract_text_with_unstructured(tmp_file_path)
            structured_text = preprocess_text(extracted_text)
            
            metadata = {
                "filename": uploaded_file.name,
                "file_type": Path(uploaded_file.name).suffix[1:],
                "timestamp": str(datetime.now())
            }
            documents = self.create_documents(structured_text, metadata)
            
            try:
                index = VectorStoreIndex.from_documents(
                    documents,
                    storage_context=self.storage_context
                )
                st.session_state['index'] = index
            except Exception as e:
                st.warning(f"Warning: Vector indexing failed - {str(e)}")
            
            return structured_text
        finally:
            os.unlink(tmp_file_path)
    
    def extract_resume_details(self, text: str) -> dict:
        """Extract resume details with improved section recognition"""
        sections = {
            'education': [],
            'experience': [],
            'skills': [],
            'projects': [],
            'certifications': []
        }
        
        current_section = None
        lines = text.split('\n')
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
                
            for section in sections.keys():
                if re.search(f"^{section}:", line, re.IGNORECASE):
                    current_section = section
                    break
            
            if current_section and line and not line.endswith(':'):
                sections[current_section].append(line)
        
        return sections

    def generate_cover_letter(self, resume_text: str, job_description: str, header_info: dict) -> str:
        """Generate a cover letter with streaming and header information"""
        try:
            prompt = COVER_LETTER_PROMPT.format(
                resume_context=resume_text,
                job_description=job_description
            )
            
            # Create header markdown
            header_md = f"""
<div style="text-align: center; margin-bottom: 20px;">
<h2>{header_info['name']}</h2>
<p>{header_info['contact']}</p>
<p>{header_info['location']}</p>
<p>{header_info['links']}</p>
</div>

---
"""
            # Display header
            st.markdown(header_md, unsafe_allow_html=True)
            
            # Create placeholder for streaming output
            stream_placeholder = st.empty()
            generated_text = ""
            
            # Stream the response
            for response in Settings.llm.stream_complete(prompt):
                chunk = response.delta
                generated_text += chunk
                # Update the placeholder with the accumulated text
                stream_placeholder.markdown(generated_text)
            
            # Store the full text in session state
            st.session_state['generated_text'] = generated_text
            st.session_state['header_info'] = header_info
            
            return generated_text
            
        except Exception as e:
            st.error(f"Error generating cover letter: {str(e)}")
            return None

    def save_as_pdf(self, cover_letter: str, header_info: dict, filename="cover_letter.pdf") -> str:
        """Save cover letter as PDF using FPDF2"""
        try:
            # Create PDF object
            pdf = PDF('P', 'mm', 'Letter')
            pdf.set_margins(25.4, 25.4, 25.4)  # 1-inch margins
            pdf.add_page()
            pdf.set_auto_page_break(auto=True, margin=25.4)
            
            # Header section
            pdf.set_font('Helvetica', 'B', 14)
            pdf.cell(0, 10, header_info['name'], align='C', ln=True)
            
            pdf.set_font('Helvetica', '', 10)
            for info in [header_info['contact'], header_info['location'], header_info['links']]:
                pdf.cell(0, 5, info, align='C', ln=True)
            
            pdf.ln(10)
            
            # Date
            pdf.cell(0, 10, datetime.now().strftime("%B %d, %Y"), ln=True)
            pdf.ln(10)
            
            # Cover letter content
            pdf.set_font('Helvetica', '', 11)
            
            # Split content into paragraphs
            paragraphs = cover_letter.split('\n\n')
            
            for paragraph in paragraphs:
                if paragraph.strip():
                    # Wrap text to fit page width (accounting for margins)
                    lines = textwrap.wrap(paragraph.strip(), width=75)
                    for line in lines:
                        pdf.multi_cell(0, 5, line)
                    pdf.ln(5)
            
            # Save PDF
            pdf.output(filename)
            return filename
            
        except Exception as e:
            st.error(f"Error saving PDF: {str(e)}")
            return None

def validate_inputs(uploaded_file, job_description, header_info):
    """Validate all inputs before processing"""
    if not uploaded_file:
        st.warning("Please upload a resume first.")
        return False
    
    if not job_description.strip():
        st.warning("Please enter a job description.")
        return False
    
    # Validate header information
    required_fields = ['name', 'contact', 'location', 'links']
    for field in required_fields:
        if not header_info.get(field, '').strip():
            st.warning(f"Please enter your {field.replace('_', ' ').title()}")
            return False
    
    return True

def main():
    st.set_page_config(layout="wide")
    
    try:
        generator = EnhancedCoverLetterGenerator()
    except Exception as e:
        st.error(f"Failed to initialize generator: {str(e)}")
        return
    
    st.title("Enhanced Resume Processor")
    
    # Create two columns for input layout
    col1, col2 = st.columns([1, 1])
    
    with col1:
        # Personal Information Section
        st.subheader("Personal Information")
        header_info = {
            'name': st.text_input("Full Name"),
            'contact': st.text_input("Contact Info (Phone & Email)"),
            'location': st.text_input("Location"),
            'links': st.text_input("Professional Links (LinkedIn, Portfolio, etc.)")
        }
        
        uploaded_file = st.file_uploader(
            "Upload Resume (PDF/DOCX)", 
            type=['pdf', 'docx']
        )
        
        if uploaded_file:
            process_button = st.button("Process Resume")
            if process_button:
                with st.spinner("Processing resume..."):
                    try:
                        structured_text = generator.process_uploaded_file(uploaded_file)
                        st.session_state['resume_text'] = structured_text
                        
                        sections = generator.extract_resume_details(structured_text)
                        
                        tabs = st.tabs(["Education", "Experience", "Skills", "Projects", "Certifications"])
                        
                        for tab, (section, content) in zip(tabs, sections.items()):
                            with tab:
                                st.subheader(section.title())
                                if content:
                                    for item in content:
                                        st.write(f"• {item}")
                                else:
                                    st.write("No information found for this section.")
                        
                        st.success("Resume processed successfully!")
                        
                    except Exception as e:
                        st.error(f"Error processing resume: {str(e)}")
    
    with col2:
        st.subheader("Job Description")
        job_description = st.text_area(
            "Enter the job description for cover letter generation",
            height=300
        )
        
        # Only show generate button if resume is processed
        if 'resume_text' in st.session_state:
            generate_button = st.button("Generate Cover Letter")
            if generate_button:
                if validate_inputs(uploaded_file, job_description, header_info):
                    with st.spinner("Generating cover letter..."):
                        cover_letter = generator.generate_cover_letter(
                            st.session_state['resume_text'],
                            job_description,
                            header_info
                        )
                        
                        if cover_letter:
                            st.session_state['cover_letter'] = cover_letter
                            
                            # Add PDF download button after letter generation
                            st.markdown("### Download Options")
                            if st.button("Create PDF"):
                                with st.spinner("Creating PDF..."):
                                    pdf_path = generator.save_as_pdf(
                                        st.session_state['generated_text'],
                                        st.session_state['header_info']
                                    )
                                    if pdf_path:
                                        with open(pdf_path, "rb") as pdf_file:
                                            pdf_bytes = pdf_file.read()
                                            st.download_button(
                                                label="Download PDF",
                                                data=pdf_bytes,
                                                file_name="cover_letter.pdf",
                                                mime="application/pdf"
                                            )
        else:
            st.info("Please process a resume first before generating a cover letter.")

if __name__ == "__main__":
    main()