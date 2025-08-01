# ClauseWise - Legal Intelligence Platform

A comprehensive, hackathon-ready legal document analysis platform using Hugging Face models and IBM Granite integration.

## 🎯 **Hackathon-Ready Features**

### ✅ **1. Clause Simplification**
- **Automatic Rewriting**: Complex legal clauses → Simple, layman-friendly language
- **LLM-Powered**: Uses HF models for intelligent simplification
- **Fallback System**: Basic simplification if LLM fails
- **Clean Output**: No `\n` clutter, properly formatted responses

### ✅ **2. Named Entity Recognition (NER)**
- **Parties**: Companies, individuals, organizations
- **Dates**: Effective dates, deadlines, timeframes
- **Monetary Amounts**: Salaries, fees, payments, compensation
- **Locations**: Addresses, jurisdictions, venues
- **Legal Terms**: 25+ legal concepts and terms
- **Obligations**: Duties, responsibilities, requirements
- **Penalties**: Consequences, damages, sanctions
- **Jurisdictions**: Governing law, applicable law

### ✅ **3. Clause Extraction & Breakdown**
- **15+ Clause Types**: Confidentiality, Payment, Termination, Liability, etc.
- **Importance Scoring**: 🔴 High, 🟡 Medium, 🟢 Low priority
- **Smart Grouping**: Clauses organized by type
- **Comprehensive Coverage**: 25+ clauses per document

### ✅ **4. Document Type Classification**
- **NDA**: Non-Disclosure Agreements
- **Employment Contract**: Job agreements, terms
- **Service Agreement**: Provider-client contracts
- **Lease Agreement**: Property rental contracts
- **Purchase Agreement**: Sales contracts
- **Partnership Agreement**: Business partnerships
- **License Agreement**: Intellectual property licenses
- **General Legal Document**: Fallback classification

### ✅ **5. Multi-Format Document Support**
- **PDF**: Full text extraction with pdfplumber
- **DOCX**: Microsoft Word documents
- **TXT**: Plain text files
- **Seamless Processing**: Automatic format detection

## 🚀 **Quick Start**

1. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

2. **Setup:**
   ```bash
   python setup.py
   ```

3. **Get HF token:**
   - Go to https://huggingface.co/settings/tokens
   - Create a new token
   - Edit `.env` file and add your token

4. **Run the application:**
   ```bash
   # Terminal 1: Backend
   python main.py
   
   # Terminal 2: Frontend
   streamlit run app.py
   ```

## 🎨 **Beautiful UI Features**

- **Modern Design**: Gradient backgrounds, clean cards
- **Formatted Responses**: No clutter, proper paragraphs
- **Importance Indicators**: Color-coded clause importance
- **Entity Visualization**: Organized NER display
- **Responsive Layout**: Works on all devices
- **Interactive Elements**: Hover effects, smooth transitions

## 📁 **Project Structure**

```
├── main.py              # FastAPI backend (enhanced features)
├── app.py               # Streamlit frontend (beautiful UI)
├── config.py            # Model configuration
├── hf_model_manager.py  # API-based model manager
├── setup.py             # Quick setup script
├── requirements.txt     # Lightweight dependencies
└── .env                 # Environment variables
```

## 🔧 **Technical Features**

- **API-Based**: No heavy downloads, fast startup
- **IBM Granite Ready**: Configured for IBM models
- **Fallback System**: Multiple models with automatic switching
- **Error Handling**: Graceful degradation
- **Clean Code**: No clutter, production-ready
- **Token Security**: Private .env file

## 🎉 **Hackathon Ready!**

Your project includes all required features:
- ✅ **Clause Simplification** - Automatic rewriting
- ✅ **NER** - Comprehensive entity extraction  
- ✅ **Clause Extraction** - Smart breakdown and analysis
- ✅ **Document Classification** - Accurate categorization
- ✅ **Multi-Format Support** - PDF, DOCX, TXT
- ✅ **Beautiful UI** - Clean, formatted responses

**Ready to impress at your hackathon!** 🚀 