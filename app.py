import streamlit as st
import requests
import re
from typing import Optional
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configure page
st.set_page_config(
    page_title="ClauseWise - Legal Intelligence",
    page_icon="⚖️",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Custom CSS for beautiful UI
st.markdown("""
<style>
    /* Global styles */
    .main .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
        max-width: 1200px;
    }
    
    /* Header styling */
    .main-header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 3rem 2rem;
        border-radius: 20px;
        color: white;
        text-align: center;
        margin-bottom: 3rem;
        box-shadow: 0 10px 30px rgba(102, 126, 234, 0.3);
    }
    
    .main-header h1 {
        font-size: 3rem;
        font-weight: 700;
        margin-bottom: 0.5rem;
    }
    
    .main-header p {
        font-size: 1.1rem;
        margin: 0.5rem 0;
        opacity: 0.9;
    }
    
    /* Option selection styling */
    .stRadio > div {
        background: rgba(255, 255, 255, 0.05);
        border-radius: 15px;
        padding: 1.5rem;
        margin: 1rem 0;
        border: 1px solid rgba(102, 126, 234, 0.2);
    }
    
    .stRadio > div > label {
        font-size: 1.1rem;
        font-weight: 600;
        color: #333;
    }
    

    

    
    /* Result cards */
    .result-card {
        background: rgba(255, 255, 255, 0.98);
        border-radius: 15px;
        padding: 2.5rem;
        margin: 2rem 0;
        box-shadow: 0 10px 30px rgba(0, 0, 0, 0.1);
        border-left: 5px solid #667eea;
    }
    
    /* Button styling */
    .stButton > button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        border-radius: 50px;
        padding: 1rem 3rem;
        font-weight: 600;
        font-size: 1.1rem;
        transition: all 0.3s ease;
        box-shadow: 0 5px 15px rgba(102, 126, 234, 0.3);
    }
    
    .stButton > button:hover {
        transform: translateY(-3px);
        box-shadow: 0 10px 25px rgba(102, 126, 234, 0.4);
        background: linear-gradient(135deg, #5a6fd8 0%, #6a4190 100%);
    }
    
    /* Text area styling */
    .stTextArea > div > div > textarea {
        border-radius: 10px;
        border: 2px solid #e0e4ff;
        padding: 1rem;
        font-size: 1rem;
        transition: all 0.3s ease;
    }
    
    .stTextArea > div > div > textarea:focus {
        border-color: #667eea;
        box-shadow: 0 0 0 3px rgba(102, 126, 234, 0.1);
    }
    
    /* Source links */
    .source-link {
        color: #667eea;
        text-decoration: none;
        padding: 0.5rem 1rem;
        border-radius: 25px;
        background: linear-gradient(135deg, #f0f2ff 0%, #e8ecff 100%);
        margin: 0.3rem;
        display: inline-block;
        font-weight: 500;
        transition: all 0.3s ease;
        border: 1px solid rgba(102, 126, 234, 0.2);
    }
    
    .source-link:hover {
        background: linear-gradient(135deg, #e0e4ff 0%, #d8dcff 100%);
        transform: translateY(-1px);
        box-shadow: 0 5px 15px rgba(102, 126, 234, 0.2);
        text-decoration: none;
    }
    
    /* Content boxes */
    .summary-box {
        background: linear-gradient(135deg, #f8f9ff 0%, #f0f2ff 100%);
        border-radius: 12px;
        padding: 2rem;
        margin: 1rem 0;
        border-left: 4px solid #667eea;
        white-space: pre-wrap;
        line-height: 1.7;
        font-size: 1rem;
        max-height: 400px;
        overflow-y: auto;
        box-shadow: 0 3px 10px rgba(0, 0, 0, 0.05);
    }
    
    /* Clause styling */
    .clause-original {
        background: linear-gradient(135deg, #f8f9ff 0%, #f0f2ff 100%);
        border-radius: 10px;
        padding: 1.5rem;
        margin: 1rem 0;
        border-left: 4px solid #667eea;
        white-space: pre-wrap;
        line-height: 1.6;
        font-size: 0.95rem;
        box-shadow: 0 2px 8px rgba(0, 0, 0, 0.05);
    }
    
    .clause-simplified {
        background: linear-gradient(135deg, #f0f8ff 0%, #e8f4ff 100%);
        border-radius: 10px;
        padding: 1.5rem;
        margin: 1rem 0;
        border-left: 4px solid #28a745;
        white-space: pre-wrap;
        line-height: 1.6;
        font-size: 0.95rem;
        box-shadow: 0 2px 8px rgba(0, 0, 0, 0.05);
    }
    
    /* Section headers */
    .section-header {
        font-size: 1.5rem;
        font-weight: 700;
        color: #ffffff;
        margin: 2rem 0 1rem 0;
        padding-bottom: 0.5rem;
        border-bottom: 2px solid #667eea;
        text-shadow: 0 1px 2px rgba(0, 0, 0, 0.3);
    }
    
    /* Success/info boxes */
    .stSuccess {
        background: linear-gradient(135deg, #d4edda 0%, #c3e6cb 100%);
        border-radius: 10px;
        padding: 1rem 1.5rem;
        border-left: 4px solid #28a745;
        margin: 1rem 0;
    }
    
    .stInfo {
        background: linear-gradient(135deg, #d1ecf1 0%, #bee5eb 100%);
        border-radius: 10px;
        padding: 1rem 1.5rem;
        border-left: 4px solid #17a2b8;
        margin: 1rem 0;
    }
    
    /* Expander styling */
    .streamlit-expanderHeader {
        background: linear-gradient(135deg, #f8f9ff 0%, #f0f2ff 100%);
        border-radius: 8px;
        padding: 1rem 1.5rem;
        font-weight: 600;
        color: #333;
        border: 1px solid rgba(102, 126, 234, 0.2);
    }
    
    .streamlit-expanderContent {
        padding: 1.5rem;
        background: rgba(255, 255, 255, 0.5);
        border-radius: 0 0 8px 8px;
        margin-top: -1px;
    }
    
    /* Responsive design */
    @media (max-width: 768px) {
        .main-header {
            padding: 2rem 1rem;
        }
        
        .main-header h1 {
            font-size: 2rem;
        }
        
        .content-card {
            padding: 1.5rem;
        }
        
        .result-card {
            padding: 1.5rem;
        }
    }
</style>
""", unsafe_allow_html=True)

API_URL = os.getenv("API_URL", "http://127.0.0.1:8001")

def main():
    # Header
    st.markdown("""
    <div class="main-header">
        <h1>⚖️ ClauseWise</h1>
        <p style="font-size: 1.2rem; margin: 0;">Legal Intelligence Platform</p>
        <p style="font-size: 0.9rem; margin: 0; opacity: 0.8;">Ask questions • Upload documents • Get insights</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Welcome message
    st.info("🚀 Ready to help! Upload a legal document or ask a question to get started.")
    
    # Two main options
    option = st.radio(
        "Choose your option:",
        ["📄 Upload & Analyze Document", "🔍 Ask Legal Questions"],
        horizontal=True
    )
    
    if option == "📄 Upload & Analyze Document":
        upload_document_page()
    else:
        ask_questions_page()

def ask_questions_page():
    st.markdown('<div class="section-header">🔍 Ask Legal Questions</div>', unsafe_allow_html=True)
    st.markdown("Ask questions about laws, policies, or legal concepts. Get answers from online sources.")
    
    question = st.text_area(
        "Enter your legal question:",
        placeholder="e.g., 'What are the key obligations in an NDA?' or 'Explain the termination clause in employment contracts'",
        height=100
    )
    
    if st.button("🔍 Search & Answer", use_container_width=True):
        if question.strip():
            with st.spinner("🔍 Searching and analyzing..."):
                try:
                    response = requests.post(f"{API_URL}/ask", json={"question": question})
                    if response.status_code == 200:
                        data = response.json()
                        display_search_results(data, question)
                    else:
                        st.error(f"Error: {response.status_code} - {response.text}")
                except Exception as e:
                    st.error(f"❌ Error: {str(e)}")
        else:
            st.warning("⚠️ Please enter a question.")

def upload_document_page():
    st.markdown('<div class="section-header">📄 Upload & Analyze Document</div>', unsafe_allow_html=True)
    st.markdown("Upload a legal document for detailed analysis and ask questions about it.")
    
    # File upload
    uploaded_file = st.file_uploader(
        "Upload your legal document",
        type=['pdf', 'docx', 'txt'],
        help="Supported formats: PDF, DOCX, TXT"
    )
    
    if uploaded_file:
        # Document analysis
        if st.button("📊 Analyze Document", use_container_width=True):
            with st.spinner("📊 Analyzing document..."):
                try:
                    files = {"file": uploaded_file}
                    data = {"analysis_type": "all"}
                    response = requests.post(f"{API_URL}/analyze-document", files=files, data=data)
                    if response.status_code == 200:
                        results = response.json()
                        display_document_summary(results, uploaded_file)
                    else:
                        st.error(f"Error: {response.status_code} - {response.text}")
                except Exception as e:
                    st.error(f"❌ Error: {str(e)}")
        
        # Ask questions about the uploaded document
        st.markdown('<div class="section-header">💭 Ask Questions About This Document</div>', unsafe_allow_html=True)
        
        doc_question = st.text_area(
            "Ask a question about the uploaded document:",
            placeholder="e.g., 'What are the main obligations?' or 'What happens if there's a breach?'",
            height=80
        )
        
        if st.button("🔍 Ask About Document", use_container_width=True):
            if doc_question.strip():
                with st.spinner("🔍 Analyzing document and answering..."):
                    try:
                        files = {"file": uploaded_file}
                        data = {
                            "question": doc_question,
                            "include_document_analysis": True
                        }
                        response = requests.post(f"{API_URL}/search-and-analyze", files=files, data=data)
                        if response.status_code == 200:
                            results = response.json()
                            display_document_qa_results(results, doc_question, uploaded_file)
                        else:
                            st.error(f"Error: {response.status_code} - {response.text}")
                    except Exception as e:
                        st.error(f"❌ Error: {str(e)}")
            else:
                st.warning("⚠️ Please enter a question about the document.")

def display_search_results(data: dict, question: str):
    st.markdown("## 🔍 Comprehensive Legal Analysis")
    
    answer = data.get("answer", "No answer returned.")
    sources = data.get("sources", [])
    
    # Clean up the answer text - replace \n with proper line breaks
    if isinstance(answer, str):
        # Handle different types of newline characters
        answer = answer.replace('\\n', '\n').replace('\n\n\n', '\n\n')
        # Format the answer properly
        answer = answer.strip()
    
    # Display question
    st.markdown(f"**❓ Question:** {question}")
    st.markdown("---")
    
    # Display comprehensive answer
    st.markdown("**💡 Detailed Answer:**")
    st.markdown(answer)
    
    # Display comprehensive sources
    if sources:
        st.markdown("---")
        st.markdown("**📚 Comprehensive Legal Sources:**")
        
        # Group sources by type
        primary_sources = []
        secondary_sources = []
        case_law = []
        general_sources = []
        
        for source in sources:
            source_lower = source.lower()
            if any(word in source_lower for word in ['act', 'statute', 'code']):
                primary_sources.append(source)
            elif any(word in source_lower for word in ['court', 'judgment', 'case']):
                case_law.append(source)
            elif any(word in source_lower for word in ['commission', 'authority', 'board']):
                secondary_sources.append(source)
            else:
                general_sources.append(source)
        
        # Display sources as clickable links
        if primary_sources:
            st.markdown("**📖 Primary Legislation:**")
            for i, src in enumerate(primary_sources, 1):
                # Create clickable link
                if any(word in src.lower() for word in ['act', 'code', 'statute']):
                    link = f"https://legislative.gov.in/{src.lower().replace(' ', '-')}"
                else:
                    link = f"https://www.google.com/search?q={src.replace(' ', '+')}"
                st.markdown(f"{i}. [{src}]({link})")
            st.markdown("")
        
        if case_law:
            st.markdown("**⚖️ Case Law & Judgments:**")
            for i, src in enumerate(case_law, 1):
                link = f"https://indiankanoon.org/search/?formInput={src.replace(' ', '+')}"
                st.markdown(f"{i}. [{src}]({link})")
            st.markdown("")
        
        if secondary_sources:
            st.markdown("**🏛️ Regulatory Bodies:**")
            for i, src in enumerate(secondary_sources, 1):
                link = f"https://www.google.com/search?q={src.replace(' ', '+')}+official+website"
                st.markdown(f"{i}. [{src}]({link})")
            st.markdown("")
        
        if general_sources:
            st.markdown("**📚 General References:**")
            for i, src in enumerate(general_sources, 1):
                link = f"https://www.google.com/search?q={src.replace(' ', '+')}"
                st.markdown(f"{i}. [{src}]({link})")
        
        # If sources are URLs (from Google search), display them directly
        if not any([primary_sources, case_law, secondary_sources, general_sources]) and sources:
            st.markdown("**🔍 Research Sources:**")
            for i, src in enumerate(sources, 1):
                if src.startswith('http'):
                    # Extract domain name for display
                    from urllib.parse import urlparse
                    try:
                        domain = urlparse(src).netloc
                        display_name = domain.replace('www.', '').replace('.gov.in', ' (Official)').replace('.org', '').replace('.in', '')
                        st.markdown(f"{i}. [{display_name}]({src})")
                    except:
                        st.markdown(f"{i}. [{src}]({src})")
                else:
                    st.markdown(f"{i}. {src}")
        
        # If no sources found, provide a note
        if not sources:
            st.markdown("**Note:** No specific sources found for this question. Answer is based on general Indian legal knowledge.")

def display_document_summary(data: dict, uploaded_file):
    st.markdown(f"## 📄 Document Analysis: {uploaded_file.name}")
    
    # Document type classification
    if "document_type" in data:
        st.markdown(f"**📋 Document Type:** {data['document_type']}")
        st.markdown("---")
    
    # Simple document summary in layman language
    if "clauses" in data and data["clauses"]:
        st.markdown("### 📋 Document Summary")
        
        # Extract key information from all clauses
        all_amounts = []
        all_dates = []
        all_parties = []
        all_obligations = []
        all_types = []
        for clause in data["clauses"]:
            original = clause.get("original", "")
            clause_type = clause.get("type", "General")
            all_types.append(clause_type)
            # Extract amounts
            amount_patterns = [
                r'(?:rs\.?|₹|rupees?|inr)\s*[:\-]?\s*(\d+(?:,\d{3})*(?:\.\d{2})?)',
                r'(\d+(?:,\d{3})*(?:\.\d{2})?)\s*(?:rs\.?|₹|rupees?|inr)',
                r'(\d+(?:,\d{3})*(?:\.\d{2})?)\s*(?:rupees?|rs?|inr)',
                r'(\d+(?:,\d{3})*(?:\.\d{2})?)\s*(?:lakh|lacs)',
                r'(\d+(?:,\d{3})*(?:\.\d{2})?)\s*(?:crore|crores)',
            ]
            for pattern in amount_patterns:
                found = re.findall(pattern, original, re.IGNORECASE)
                if found:
                    all_amounts.extend(found)
            # Extract dates
            dates = re.findall(r'\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b|\b\d{4}\b|\b(?:january|february|march|april|may|june|july|august|september|october|november|december)\s+\d{1,2},?\s+\d{4}\b', original, re.IGNORECASE)
            all_dates.extend(dates)
            # Extract parties
            parties = re.findall(r'\b(?:party|parties|company|corporation|individual|employer|employee|landlord|tenant|buyer|seller|lessor|lessee)\b', original, re.IGNORECASE)
            all_parties.extend(parties)
            # Extract obligations
            obligations = re.findall(r'\b(?:shall|will|must|agrees?|acknowledges?|represents?|warrants?|obligated|responsible)\b', original, re.IGNORECASE)
            all_obligations.extend(obligations)
        # Compose a layman summary
        summary_points = []
        if all_types:
            summary_points.append(f"This document covers: {', '.join(sorted(set(all_types)))} clauses.")
        if all_parties:
            summary_points.append(f"The main parties involved are: {', '.join(sorted(set([p.title() for p in all_parties]))) }.")
        if all_amounts:
            summary_points.append(f"Key amounts mentioned include: {', '.join(['₹'+a.replace(',', '') for a in sorted(set(all_amounts))]) }.")
        if all_dates:
            summary_points.append(f"Important dates in the document: {', '.join(sorted(set(all_dates)))}.")
        if all_obligations:
            summary_points.append(f"The document sets out obligations such as: {', '.join(sorted(set([o.title() for o in all_obligations]))) }.")
        # Limit to 3-5 points
        for point in summary_points[:5]:
            st.markdown(f"- {point}")
        st.markdown("---")
    
    # Enhanced NER display
    if "entities" in data:
        st.markdown("### 🏷️ Named Entity Recognition")
        
        entities = data["entities"]
        
        # Create columns for better layout
        col1, col2 = st.columns(2)
        
        with col1:
            if entities.get("parties"):
                st.markdown("**👥 Parties:**")
                for party in entities["parties"][:5]:
                    st.markdown(f"• {party}")
            
            if entities.get("dates"):
                st.markdown("**📅 Dates:**")
                for date in entities["dates"][:5]:
                    st.markdown(f"• {date}")
            
            if entities.get("amounts"):
                st.markdown("**💰 Monetary Amounts:**")
                for amount in entities["amounts"][:5]:
                    # Ensure amount is properly formatted
                    if isinstance(amount, str):
                        # Clean up the amount display
                        clean_amount = amount.replace(',', '')
                        if not clean_amount.startswith('₹'):
                            clean_amount = f"₹{clean_amount}"
                        st.markdown(f"• {clean_amount}")
                    else:
                        st.markdown(f"• {amount}")
        
        with col2:
            if entities.get("locations"):
                st.markdown("**📍 Locations:**")
                for location in entities["locations"][:5]:
                    st.markdown(f"• {location}")
            
            if entities.get("legal_terms"):
                st.markdown("**⚖️ Legal Terms:**")
                for term in entities["legal_terms"][:5]:
                    st.markdown(f"• {term}")
            
            if entities.get("obligations"):
                st.markdown("**📋 Obligations:**")
                for obligation in entities["obligations"][:3]:
                    st.markdown(f"• {obligation[:100]}{'...' if len(obligation) > 100 else ''}")
        
        # Show additional entity types if available
        if entities.get("penalties") or entities.get("jurisdictions"):
            st.markdown("**🔍 Additional Entities:**")
            if entities.get("penalties"):
                st.markdown("*Penalties:* " + ", ".join(entities["penalties"][:3]))
            if entities.get("jurisdictions"):
                st.markdown("*Jurisdictions:* " + ", ".join(entities["jurisdictions"][:3]))

def display_document_qa_results(data: dict, question: str, uploaded_file):
    st.markdown("## 💭 Document-Specific Legal Analysis")
    
    # Document-based answer
    if "document_analysis" in data:
        # Display question and document info
        col1, col2 = st.columns(2)
        with col1:
            st.markdown(f"**❓ Question:** {question}")
        with col2:
            st.markdown(f"**📄 Document:** {uploaded_file.name}")
        
        st.markdown("---")
        
        # Display document-based answer if available
        if "document_answer" in data:
            answer = data["document_answer"]
            if isinstance(answer, dict) and "result" in answer:
                answer_text = answer["result"]
            elif isinstance(answer, str):
                answer_text = answer
            else:
                answer_text = str(answer)
            
            # Clean up the answer text
            if isinstance(answer_text, str):
                answer_text = answer_text.replace('\\n', '\n').replace('\n\n\n', '\n\n').strip()
            
            st.markdown("**💡 Document-Based Answer:**")
            st.markdown(answer_text)
            st.markdown("---")
        
        # Simple document summary
        doc_type = data["document_analysis"].get("document_type", "Unknown")
        st.markdown(f"**Document Type:** {doc_type}")

if __name__ == "__main__":
    main() 