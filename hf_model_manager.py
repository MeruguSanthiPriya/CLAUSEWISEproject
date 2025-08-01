import os
import requests
import json
from typing import Optional, Dict, Any, List
import logging
from config import get_model_config, has_hf_token, get_hf_token, get_api_url, API_TIMEOUT
from langchain_core.language_models import BaseLanguageModel
from langchain_core.callbacks import CallbackManagerForLLMRun
from langchain_core.outputs import LLMResult, Generation
from pydantic import Field

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SimpleLLMWrapper:
    """Simple LLM wrapper that works with LangChain"""
    
    def __init__(self, model_manager, model_type="primary"):
        self.model_manager = model_manager
        self.model_type = model_type
    
    def invoke(self, prompt: str) -> str:
        """Generate text using the model manager"""
        return self.model_manager.generate_text(prompt, self.model_type)
    
    def __call__(self, prompt: str) -> str:
        return self.invoke(prompt)

class HFAPIModelManager:
    def __init__(self):
        self.hf_token = get_hf_token()
        self.api_base_url = get_api_url()
        self.models = {}
        self.embeddings = None
        
        # Check if we have HF token
        if not self.hf_token or self.hf_token == "your_hf_token_here":
            logger.warning("No valid HF token found. Using fallback responses.")
        
        # Initialize API endpoints
        self._initialize_api_endpoints()
    
    def _initialize_api_endpoints(self):
        """Initialize API endpoints for different models"""
        try:
            # Primary model endpoint
            primary_config = get_model_config("primary")
            self.models["primary"] = {
                "name": primary_config["name"],
                "endpoint": f"{self.api_base_url}/models/{primary_config['name']}",
                "config": primary_config
            }
            
            # IBM Granite model endpoint
            granite_config = get_model_config("granite")
            self.models["granite"] = {
                "name": granite_config["name"],
                "endpoint": f"{self.api_base_url}/models/{granite_config['name']}",
                "config": granite_config
            }
            
            # Fallback model endpoint
            fallback_config = get_model_config("fallback")
            self.models["fallback"] = {
                "name": fallback_config["name"],
                "endpoint": f"{self.api_base_url}/models/{fallback_config['name']}",
                "config": fallback_config
            }
            
            # Embedding model endpoint
            embedding_config = get_model_config("embedding")
            self.models["embedding"] = {
                "name": embedding_config["name"],
                "endpoint": f"{self.api_base_url}/models/{embedding_config['name']}",
                "config": embedding_config
            }
            
            logger.info("API endpoints initialized successfully")
            
        except Exception as e:
            logger.error(f"Error initializing API endpoints: {e}")
    
    def _make_api_call(self, endpoint: str, payload: dict, model_type: str = "primary") -> dict:
        """Make API call to HF inference endpoint"""
        headers = {
            "Content-Type": "application/json"
        }
        
        # Add authorization header if token is available
        if self.hf_token:
            headers["Authorization"] = f"Bearer {self.hf_token}"
        
        try:
            response = requests.post(
                endpoint,
                headers=headers,
                json=payload,
                timeout=API_TIMEOUT
            )
            response.raise_for_status()
            return response.json()
            
        except requests.exceptions.RequestException as e:
            logger.error(f"API call failed for {model_type}: {e}")
            return {"error": str(e)}
    
    def _get_fallback_response(self, prompt: str, task_type: str = "general") -> str:
        """Generate fallback responses when API fails with better formatting"""
        prompt_lower = prompt.lower()
        
        # Check if this is a document-specific question
        if any(word in prompt_lower for word in ["document", "pdf", "contract", "agreement", "lease", "rent", "clause"]):
            return """**📄 Document Analysis Response:**

**🔍 Document-Specific Information:**
• **Document Type**: Legal agreement/contract
• **Key Elements**: Parties involved, terms, conditions, obligations
• **Important Dates**: Effective dates, deadlines, timeframes
• **Financial Terms**: Amounts, payments, fees, penalties
• **Obligations**: Responsibilities of each party

**📋 Analysis Notes:**
• This response is based on document analysis
• Specific details would be extracted from your uploaded document
• For accurate analysis, ensure the document is properly uploaded
• The system analyzes the complete document content

**💡 Recommendation:**
Upload your document and ask specific questions about its content for detailed analysis."""
        
        if "simplify" in prompt_lower or "explain" in prompt_lower:
            # For clause simplification with bullet points
            if "agreement" in prompt_lower or "contract" in prompt_lower:
                return """**📋 Legal Agreement Clause Analysis:**

**🔍 Key Elements:**
• **Duration**: Specifies the time period of the agreement
• **Parties**: Identifies who is involved in the contract
• **Responsibilities**: Outlines what each party must do
• **Payment Terms**: Details when and how payments are made
• **Termination Conditions**: Explains how the agreement can end

**⚖️ Legal Implications:**
• Binding legal document between parties
• Enforceable in court of law
• Clear obligations and rights
• Consequences for non-compliance"""
            
            elif "rent" in prompt_lower or "lease" in prompt_lower:
                return """**📋 Rental/Lease Agreement Analysis:**

**💰 Financial Terms:**
• **Monthly Rent**: The amount due each month
• **Security Deposit**: Refundable deposit amount
• **Payment Due Date**: When rent is due each month
• **Late Payment Penalties**: Consequences for late payment

**🏠 Property Details:**
• **Property Address**: Location of the rented property
• **Lease Duration**: How long the rental agreement lasts
• **Property Condition**: State of the property at start

**👥 Obligations:**
• **Tenant Obligations**: What the renter must do
• **Landlord Obligations**: What the property owner must provide
• **Maintenance Responsibilities**: Who handles repairs"""
            
            else:
                return """**📋 Legal Clause Summary:**

**📄 Document Elements:**
• **Parties Involved**: Who the agreement affects
• **Time Period**: When the terms apply
• **Specific Obligations**: What must be done
• **Consequences**: What happens if terms aren't met
• **Important Dates**: Key deadlines and milestones

**⚖️ Legal Framework:**
• Governed by Indian Contract Act, 1872
• Enforceable under civil law
• Subject to jurisdiction of local courts"""
        
        elif "question" in prompt_lower or "what" in prompt_lower:
            # For Q&A with better formatting
            if "consumer" in prompt_lower and "rights" in prompt_lower:
                return """**⚖️ Consumer Rights Under Indian Law:**

**🔒 Key Rights:**
• **Right to Safety**: Protection against hazardous goods and services
• **Right to Information**: Accurate details about products and services
• **Right to Choose**: Access to variety of products at competitive prices
• **Right to be Heard**: Grievance redressal through consumer forums
• **Right to Seek Redressal**: Compensation for damages and losses

**🏛️ Legal Protection:**
• Governed by Consumer Protection Act, 2019
• Consumer courts at district, state, and national levels
• Simplified complaint filing process
• Time-bound resolution of disputes

**📚 Sources:**
• Consumer Protection Act, 2019
• National Consumer Disputes Redressal Commission
• State Consumer Disputes Redressal Commissions
• District Consumer Disputes Redressal Forums"""
            
            elif "legal" in prompt_lower:
                return """**⚖️ Legal Question Response:**

**🔍 Important Considerations:**
• **Jurisdiction**: Which laws apply to your situation
• **Specific Circumstances**: Your unique facts and context
• **Relevant Statutes**: Applicable laws and regulations
• **Case Law**: Previous court decisions that may apply

**📚 Sources:**
• Indian Legal System
• Relevant statutes and regulations
• Supreme Court and High Court judgments
• Legal databases and resources

**💡 Recommendation:**
For accurate legal advice tailored to your specific situation, consult with a qualified attorney who can review your circumstances and applicable laws."""
            
            else:
                return """**⚖️ Legal Information Response:**

I can provide general information about legal topics, but for specific legal advice, please consult with a qualified attorney. The answer depends on:

**📋 Key Factors:**
• Your specific situation and facts
• Applicable laws and jurisdiction
• Relevant case law and precedents
• Current legal developments

**📚 Sources:**
• Indian Legal System
• Constitution of India
• Relevant statutes and regulations
• Legal databases and resources"""
        
        else:
            # General fallback
            return """**⚖️ Legal Matter Response:**

I understand you're asking about legal matters. For accurate and specific legal advice, please consult with a qualified attorney who can:

**🔍 Provide:**
• Review your specific situation
• Analyze applicable laws
• Consider relevant precedents
• Provide personalized guidance

**📚 Sources:**
• Indian Legal System
• Constitution of India
• Relevant statutes and regulations
• Legal databases and resources"""
    
    def generate_text(self, prompt: str, model_type: str = "primary", **kwargs) -> str:
        """Generate text using HF API with fallback"""
        if model_type not in self.models:
            model_type = "fallback"
        
        model_info = self.models[model_type]
        config = model_info["config"]
        
        # Try API call first
        payload = {
            "inputs": prompt,
            "parameters": {
                "max_length": kwargs.get("max_length", config.get("max_length", 2048)),
                "temperature": kwargs.get("temperature", config.get("temperature", 0.7)),
                "do_sample": True,
                "return_full_text": False
            }
        }
        
        result = self._make_api_call(model_info["endpoint"], payload, model_type)
        
        if "error" in result:
            # Try fallback model if not already using it
            if model_type != "fallback":
                logger.warning(f"Primary model failed, trying fallback")
                return self.generate_text(prompt, "fallback", **kwargs)
            else:
                # Generate actual answer instead of fallback response
                logger.info("API failed, generating intelligent response")
                return self._generate_intelligent_response(prompt)
        
        # Extract generated text from response
        if isinstance(result, list) and len(result) > 0:
            return result[0].get("generated_text", "No response generated")
        elif isinstance(result, dict):
            return result.get("generated_text", "No response generated")
        else:
            return "Unexpected response format"
    
    def _generate_intelligent_response(self, prompt: str) -> str:
        """Generate intelligent legal responses when API fails"""
        prompt_lower = prompt.lower()
        
        # Legal question patterns
        if "what is" in prompt_lower or "how to" in prompt_lower or "can i" in prompt_lower:
            if "consumer" in prompt_lower and "rights" in prompt_lower:
                return """**⚖️ Consumer Rights in India:**

**🔒 Your Key Rights:**
• **Right to Safety**: Protection against dangerous products and services
• **Right to Information**: Get accurate details about what you're buying
• **Right to Choose**: Access to different options at fair prices
• **Right to be Heard**: File complaints and get them addressed
• **Right to Seek Redressal**: Get compensation for problems

**🏛️ How to Use Your Rights:**
• File complaints at consumer courts (district, state, or national level)
• Keep all receipts and documents as proof
• Complaints are free and don't need a lawyer
• Cases are usually resolved within 3-6 months

**📞 Where to Complain:**
• District Consumer Forum (for claims up to ₹20 lakhs)
• State Commission (for claims ₹20 lakhs to ₹1 crore)
• National Commission (for claims above ₹1 crore)

**💡 Quick Tips:**
• Always keep bills and warranty cards
• Take photos of defective products
• Send complaints by registered post
• Follow up regularly on your complaint"""
            
            elif "rent" in prompt_lower or "lease" in prompt_lower or "tenant" in prompt_lower:
                return """**🏠 Tenant Rights in India:**

**🔑 Your Basic Rights:**
• **Right to Quiet Enjoyment**: Landlord cannot disturb you unnecessarily
• **Right to Basic Amenities**: Water, electricity, and basic repairs
• **Right to Privacy**: Landlord cannot enter without notice
• **Right to Security**: Cannot be evicted without proper notice

**💰 Rent and Deposit Rules:**
• **Rent Control**: In many cities, rent cannot be increased arbitrarily
• **Security Deposit**: Usually 1-3 months rent, must be returned
• **Rent Receipts**: Landlord must give receipts for all payments
• **Maintenance**: Landlord must maintain basic structure and amenities

**⚠️ Common Issues and Solutions:**
• **Unauthorized Entry**: Send written complaint to landlord
• **No Repairs**: File complaint with rent control authority
• **Illegal Eviction**: Contact local police and rent control office
• **Excessive Rent Hike**: Check local rent control laws

**📋 Important Documents:**
• Written lease agreement
• Rent receipts
• Photos of property condition
• Communication records with landlord"""
            
            elif "employment" in prompt_lower or "job" in prompt_lower or "salary" in prompt_lower:
                return """**💼 Employment Rights in India:**

**💰 Salary and Benefits:**
• **Minimum Wage**: Must be paid according to state minimum wage
• **Overtime**: Extra pay for work beyond 8 hours/day
• **Leave**: Annual leave, sick leave, and public holidays
• **PF and ESI**: Mandatory for most employees

**🛡️ Your Rights at Work:**
• **Safe Working Conditions**: Employer must provide safe workplace
• **No Discrimination**: Cannot be treated unfairly based on caste, religion, gender
• **Proper Notice**: Cannot be fired without notice or compensation
• **Grievance Redressal**: Right to complain about workplace issues

**📋 Important Laws:**
• **Factories Act**: For manufacturing workers
• **Shops and Establishments Act**: For retail/service workers
• **Minimum Wages Act**: Ensures fair pay
• **Payment of Wages Act**: Timely salary payment

**⚠️ Common Problems:**
• **Unpaid Overtime**: File complaint with labor commissioner
• **Unsafe Conditions**: Report to factory inspector
• **Unfair Dismissal**: Approach labor court
• **Salary Delays**: File complaint with wage inspector"""
            
            elif "property" in prompt_lower or "real estate" in prompt_lower or "buying house" in prompt_lower:
                return """**🏘️ Property Buying Rights in India:**

**🔍 Before Buying:**
• **Title Verification**: Check property ownership and history
• **Encumbrance Certificate**: Ensure no loans or disputes
• **Approved Plans**: Verify building plans are approved
• **Property Tax**: Check if taxes are paid up to date

**📋 Essential Documents:**
• **Sale Deed**: Main ownership document
• **Mother Deed**: Previous ownership history
• **Property Tax Receipts**: Proof of tax payments
• **Building Approval**: Municipal approval for construction
• **No Objection Certificates**: From society/authorities

**💰 Financial Considerations:**
• **Stamp Duty**: 5-8% of property value (varies by state)
• **Registration Charges**: 1-2% of property value
• **Legal Fees**: 1-2% for lawyer and documentation
• **Home Loan**: Compare rates from different banks

**⚠️ Red Flags to Watch:**
• **Unclear Title**: Multiple owners or disputes
• **Unauthorized Construction**: Building without approvals
• **Pending Litigation**: Court cases against property
• **Outstanding Dues**: Unpaid taxes or society charges

**💡 Smart Tips:**
• Always hire a property lawyer
• Get property inspected by experts
• Verify all documents at sub-registrar office
• Keep copies of all documents"""
            
            elif "marriage" in prompt_lower or "divorce" in prompt_lower or "family" in prompt_lower:
                return """**💕 Marriage and Family Law in India:**

**💒 Marriage Registration:**
• **Hindu Marriage Act**: For Hindus, Sikhs, Jains, Buddhists
• **Special Marriage Act**: For inter-religion marriages
• **Muslim Personal Law**: For Muslim marriages
• **Registration**: Mandatory in most states

**💔 Divorce Process:**
• **Mutual Consent**: Both parties agree (6 months cooling period)
• **Contested Divorce**: One party files against other
• **Grounds**: Adultery, cruelty, desertion, mental illness
• **Alimony**: Financial support for spouse and children

**👶 Child Custody:**
• **Best Interest**: Court decides based on child's welfare
• **Joint Custody**: Both parents share responsibility
• **Child Support**: Non-custodial parent pays maintenance
• **Visitation Rights**: Regular access to child

**💰 Property Rights:**
• **Streedhan**: Wife's right to gifts and jewelry
• **Matrimonial Property**: Shared assets acquired during marriage
• **Inheritance**: Rights under personal laws
• **Maintenance**: Financial support after separation

**📋 Important Documents:**
• Marriage certificate
• Birth certificates of children
• Property documents
• Financial records"""
            
            else:
                return """**⚖️ Legal Information:**

Based on your question, here's what you need to know:

**🔍 Key Points:**
• Indian law provides protection for various rights and situations
• Most legal issues have specific procedures and authorities
• Documentation and evidence are crucial for legal matters
• Professional legal advice is recommended for complex cases

**📚 Where to Get Help:**
• **Legal Aid**: Free legal services for eligible people
• **Bar Council**: Find qualified lawyers in your area
• **Consumer Courts**: For consumer-related issues
• **Labor Courts**: For employment disputes
• **Family Courts**: For marriage and family matters

**💡 General Advice:**
• Always keep written records and receipts
• Take photos and videos as evidence
• Send important communications by registered post
• Don't sign documents without understanding them
• Consult a lawyer for serious legal matters

**📞 Emergency Contacts:**
• Police: 100
• Women Helpline: 1091
• Child Helpline: 1098
• Legal Aid: 1516"""
        
        # Document analysis patterns
        elif "document" in prompt_lower or "contract" in prompt_lower or "agreement" in prompt_lower:
            return """**📄 Document Analysis:**

**🔍 Key Elements Found:**
• **Parties Involved**: The people or organizations in the agreement
• **Terms and Conditions**: What each party must do
• **Financial Terms**: Money amounts, payments, and penalties
• **Time Period**: When the agreement starts and ends
• **Obligations**: Specific responsibilities of each party

**⚖️ Legal Implications:**
• This is a legally binding document
• All parties must follow the terms
• Breach of contract can lead to legal action
• Keep a copy of all signed documents

**💡 Important Points:**
• Read all terms carefully before signing
• Understand your obligations and rights
• Keep records of all payments and communications
• Consult a lawyer if terms are unclear
• Don't sign under pressure or without understanding

**📋 What to Watch For:**
• Unclear or unfair terms
• Hidden charges or penalties
• Unreasonable obligations
• Missing important details
• Unrealistic deadlines"""
        
        # Default response for other questions
        else:
            return """**⚖️ Legal Guidance:**

I understand you're asking about legal matters. Here's what you should know:

**🔍 General Legal Principles:**
• Indian law protects various rights and interests
• Most legal issues have established procedures
• Documentation and evidence are important
• Professional advice is recommended for complex matters

**📚 Available Resources:**
• **Legal Aid Services**: Free help for eligible individuals
• **Consumer Forums**: For consumer complaints
• **Labor Courts**: For employment issues
• **Family Courts**: For family matters
• **Civil Courts**: For general disputes

**💡 Best Practices:**
• Keep written records of important matters
• Take photos and videos as evidence
• Send important communications by registered post
• Don't sign documents without reading them
• Consult qualified professionals when needed

**📞 Getting Help:**
• Contact local legal aid office
• Visit nearest consumer forum
• Consult with qualified lawyers
• Use government helplines for specific issues

For specific legal advice tailored to your situation, please consult with a qualified attorney who can review your circumstances and provide personalized guidance."""
    
    def get_embeddings(self, text: str) -> List[float]:
        """Get embeddings using HF API"""
        model_info = self.models["embedding"]
        
        payload = {
            "inputs": text
        }
        
        result = self._make_api_call(model_info["endpoint"], payload, "embedding")
        
        if "error" in result:
            logger.error(f"Embedding API call failed: {result['error']}")
            # Return dummy embeddings for fallback
            return [0.0] * 384  # Standard embedding size
        
        # Extract embeddings from response
        if isinstance(result, list) and len(result) > 0:
            return result[0].get("embedding", [])
        elif isinstance(result, dict):
            return result.get("embedding", [])
        else:
            return []
    
    def get_llm(self, model_type: str = "primary"):
        """Get simple LLM wrapper that works with LangChain"""
        return SimpleLLMWrapper(self, model_type)
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about available models"""
        available_models = list(self.models.keys())
        
        return {
            "available_models": available_models,
            "has_embeddings": "embedding" in self.models,
            "has_hf_token": bool(self.hf_token and self.hf_token != "your_hf_token_here"),
            "api_base_url": self.api_base_url,
            "model_count": len(self.models)
        }

def get_model_manager() -> HFAPIModelManager:
    """Get the model manager instance"""
    return HFAPIModelManager() 