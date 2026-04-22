"""
AutoStream Agent - Social-to-Lead Agentic Workflow
Built with LangGraph for state management and agentic control flow
"""

import json
import os
from typing import Any, Dict, List, Optional, Literal
from dataclasses import dataclass, field
from datetime import datetime

from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, SystemMessage
from langchain_core.language_model import BaseLangModel
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, END
from langgraph.types import StreamWriter
from typing_extensions import TypedDict
import re


# ============================================================================
# Data Models
# ============================================================================

class AgentState(TypedDict):
    """State management for multi-turn conversation"""
    messages: List[BaseMessage]
    intent: Optional[str]
    user_name: Optional[str]
    user_email: Optional[str]
    user_platform: Optional[str]
    lead_captured: bool
    conversation_turn: int
    knowledge_context: Optional[str]


@dataclass
class LeadInfo:
    """Data class for lead information"""
    name: Optional[str] = None
    email: Optional[str] = None
    platform: Optional[str] = None
    
    def is_complete(self) -> bool:
        return all([self.name, self.email, self.platform])


# ============================================================================
# Knowledge Base Loader
# ============================================================================

class KnowledgeBaseManager:
    """Manages loading and querying the AutoStream knowledge base"""
    
    def __init__(self, kb_path: str = "knowledge_base/autostream_kb.json"):
        self.kb_path = kb_path
        self.knowledge_base = self._load_kb()
    
    def _load_kb(self) -> Dict[str, Any]:
        """Load knowledge base from JSON file"""
        try:
            with open(self.kb_path, 'r') as f:
                return json.load(f)
        except FileNotFoundError:
            raise FileNotFoundError(f"Knowledge base not found at {self.kb_path}")
    
    def get_pricing_info(self) -> str:
        """Retrieve pricing information"""
        pricing = self.knowledge_base.get("pricing", {})
        return json.dumps(pricing, indent=2)
    
    def get_features(self) -> str:
        """Retrieve features information"""
        features = self.knowledge_base.get("features", {})
        return json.dumps(features, indent=2)
    
    def get_policies(self) -> str:
        """Retrieve company policies"""
        policies = self.knowledge_base.get("policies", {})
        return json.dumps(policies, indent=2)
    
    def get_use_cases(self) -> str:
        """Retrieve use case information"""
        use_cases = self.knowledge_base.get("use_cases", {})
        return json.dumps(use_cases, indent=2)
    
    def query_rag(self, query: str) -> str:
        """
        Simple RAG - retrieve relevant knowledge based on keywords
        In production, this would use vector embeddings for semantic search
        """
        query_lower = query.lower()
        context_parts = []
        
        # Check for pricing queries
        if any(word in query_lower for word in ["price", "cost", "plan", "subscription", "payment"]):
            context_parts.append(f"Pricing Information:\n{self.get_pricing_info()}")
        
        # Check for feature queries
        if any(word in query_lower for word in ["feature", "capability", "can", "support", "does"]):
            context_parts.append(f"Features:\n{self.get_features()}")
        
        # Check for policy queries
        if any(word in query_lower for word in ["refund", "cancel", "support", "policy", "guarantee"]):
            context_parts.append(f"Policies:\n{self.get_policies()}")
        
        # Check for use case queries
        if any(word in query_lower for word in ["youtube", "instagram", "tiktok", "use", "best for", "creator"]):
            context_parts.append(f"Use Cases:\n{self.get_use_cases()}")
        
        # If no specific context found, return general info
        if not context_parts:
            context_parts.append(f"General Information:\n{json.dumps(self.knowledge_base.get('company', {}), indent=2)}")
        
        return "\n\n".join(context_parts)


# ============================================================================
# Intent Detection
# ============================================================================

class IntentDetector:
    """Detects user intent from conversation"""
    
    INTENT_KEYWORDS = {
        "casual_greeting": [
            "hi", "hello", "hey", "good morning", "good afternoon", "good evening",
            "how are you", "what's up", "thanks", "thank you"
        ],
        "product_inquiry": [
            "tell me about", "how does", "what is", "explain", "pricing", "cost",
            "features", "plan", "subscription", "does it", "can it", "support"
        ],
        "high_intent_lead": [
            "ready to", "want to", "sign up", "try", "get started", "subscribe",
            "purchase", "buy", "interested in", "let's do this", "count me in",
            "where do i", "how do i", "what's next", "excited"
        ]
    }
    
    @staticmethod
    def detect_intent(message: str, conversation_history: List[BaseMessage]) -> str:
        """
        Detect intent from user message
        Returns one of: casual_greeting, product_inquiry, high_intent_lead
        """
        message_lower = message.lower()
        
        # Track conversation context
        has_product_discussion = any(
            "product" in str(msg).lower() or "feature" in str(msg).lower() or "price" in str(msg).lower()
            for msg in conversation_history if isinstance(msg, AIMessage)
        )
        
        # Check for high-intent signals
        for keyword in IntentDetector.INTENT_KEYWORDS["high_intent_lead"]:
            if keyword in message_lower:
                return "high_intent_lead"
        
        # Check for product inquiry
        for keyword in IntentDetector.INTENT_KEYWORDS["product_inquiry"]:
            if keyword in message_lower:
                return "product_inquiry"
        
        # Check for casual greeting
        for keyword in IntentDetector.INTENT_KEYWORDS["casual_greeting"]:
            if keyword in message_lower:
                return "casual_greeting"
        
        # Default based on context
        if has_product_discussion:
            return "product_inquiry"
        
        return "casual_greeting"


# ============================================================================
# Tool Execution
# ============================================================================

def mock_lead_capture(name: str, email: str, platform: str) -> Dict[str, Any]:
    """
    Mock API function to capture leads
    In production, this would send data to CRM/database
    """
    timestamp = datetime.now().isoformat()
    lead_id = f"LEAD_{timestamp.replace('-', '').replace(':', '').replace('.', '')}"
    
    result = {
        "success": True,
        "lead_id": lead_id,
        "name": name,
        "email": email,
        "platform": platform,
        "captured_at": timestamp,
        "message": f"✅ Lead captured successfully! Lead ID: {lead_id}"
    }
    
    print(f"\n{'='*60}")
    print("🎉 LEAD CAPTURE SUCCESSFUL")
    print(f"{'='*60}")
    print(f"Lead ID: {lead_id}")
    print(f"Name: {name}")
    print(f"Email: {email}")
    print(f"Platform: {platform}")
    print(f"Captured at: {timestamp}")
    print(f"{'='*60}\n")
    
    return result


# ============================================================================
# LLM Setup
# ============================================================================

def initialize_llm() -> BaseLangModel:
    """Initialize the LLM - uses Claude 3 Haiku by default"""
    # You can switch between models by changing the model parameter
    # Options: gpt-4o-mini, gemini-1.5-flash, claude-3-haiku
    
    api_key = os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        raise ValueError("ANTHROPIC_API_KEY not set. Please set your API key.")
    
    # Using Claude 3 Haiku (can be changed to gpt-4o-mini or gemini-1.5-flash)
    from langchain_anthropic import ChatAnthropic
    return ChatAnthropic(
        model="claude-3-5-haiku-20241022",
        api_key=api_key,
        temperature=0.7
    )


# ============================================================================
# Agent Nodes
# ============================================================================

class AutoStreamAgent:
    """Main agent class with LangGraph state management"""
    
    def __init__(self, kb_path: str = "knowledge_base/autostream_kb.json"):
        self.kb_manager = KnowledgeBaseManager(kb_path)
        self.intent_detector = IntentDetector()
        self.llm = initialize_llm()
        self.graph = self._build_graph()
    
    def _build_graph(self):
        """Build the LangGraph state machine"""
        workflow = StateGraph(AgentState)
        
        # Add nodes
        workflow.add_node("process_input", self.process_input_node)
        workflow.add_node("intent_detection", self.intent_detection_node)
        workflow.add_node("rag_retrieval", self.rag_retrieval_node)
        workflow.add_node("generate_response", self.generate_response_node)
        workflow.add_node("lead_qualification", self.lead_qualification_node)
        workflow.add_node("collect_lead_info", self.collect_lead_info_node)
        workflow.add_node("capture_lead", self.capture_lead_node)
        
        # Set entry point
        workflow.set_entry_point("process_input")
        
        # Add edges
        workflow.add_edge("process_input", "intent_detection")
        workflow.add_edge("intent_detection", "rag_retrieval")
        workflow.add_edge("rag_retrieval", "generate_response")
        workflow.add_conditional_edges(
            "generate_response",
            self.should_qualify_lead,
            {
                "lead_qualification": "lead_qualification",
                "end": END
            }
        )
        workflow.add_conditional_edges(
            "lead_qualification",
            self.should_collect_info,
            {
                "collect_lead_info": "collect_lead_info",
                "end": END
            }
        )
        workflow.add_conditional_edges(
            "collect_lead_info",
            self.is_lead_complete,
            {
                "capture_lead": "capture_lead",
                "end": END
            }
        )
        workflow.add_edge("capture_lead", END)
        
        return workflow.compile()
    
    def process_input_node(self, state: AgentState) -> AgentState:
        """Process and validate user input"""
        state["conversation_turn"] += 1
        return state
    
    def intent_detection_node(self, state: AgentState) -> AgentState:
        """Detect user intent"""
        if state["messages"]:
            last_message = state["messages"][-1]
            if isinstance(last_message, HumanMessage):
                intent = self.intent_detector.detect_intent(
                    last_message.content,
                    state["messages"][:-1]
                )
                state["intent"] = intent
        return state
    
    def rag_retrieval_node(self, state: AgentState) -> AgentState:
        """Retrieve relevant knowledge using RAG"""
        if state["messages"]:
            last_message = state["messages"][-1]
            if isinstance(last_message, HumanMessage):
                context = self.kb_manager.query_rag(last_message.content)
                state["knowledge_context"] = context
        return state
    
    def generate_response_node(self, state: AgentState) -> AgentState:
        """Generate agent response using LLM"""
        # Build system prompt
        system_prompt = self._build_system_prompt(state)
        
        # Build context for LLM
        messages_for_llm = [SystemMessage(content=system_prompt)]
        messages_for_llm.extend(state["messages"])
        
        # Generate response
        response = self.llm.invoke(messages_for_llm)
        
        # Add response to messages
        state["messages"].append(response)
        
        return state
    
    def lead_qualification_node(self, state: AgentState) -> AgentState:
        """Qualify if user is a high-intent lead"""
        if state["intent"] == "high_intent_lead":
            # Extract lead info if present in message
            last_message = state["messages"][-2]  # Get user message before agent response
            if isinstance(last_message, HumanMessage):
                state = self._extract_lead_info(state, last_message.content)
        return state
    
    def collect_lead_info_node(self, state: AgentState) -> AgentState:
        """Collect missing lead information"""
        # The LLM already generated a response asking for info
        # This node just marks that we're collecting info
        return state
    
    def capture_lead_node(self, state: AgentState) -> AgentState:
        """Capture the lead using mock API"""
        if state["user_name"] and state["user_email"] and state["user_platform"]:
            result = mock_lead_capture(
                state["user_name"],
                state["user_email"],
                state["user_platform"]
            )
            state["lead_captured"] = True
            
            # Add success message to conversation
            success_msg = AIMessage(
                content=f"🎉 Perfect! I've registered you for AutoStream Pro. "
                f"Your lead has been captured (ID: {result['lead_id']}). "
                f"You should receive a confirmation email at {state['user_email']} shortly. "
                f"Welcome to AutoStream, {state['user_name']}!"
            )
            state["messages"].append(success_msg)
        
        return state
    
    def should_qualify_lead(self, state: AgentState) -> Literal["lead_qualification", "end"]:
        """Determine if we should qualify the user as a lead"""
        if state["intent"] == "high_intent_lead" and not state["lead_captured"]:
            return "lead_qualification"
        return "end"
    
    def should_collect_info(self, state: AgentState) -> Literal["collect_lead_info", "end"]:
        """Determine if we need to collect more info"""
        lead = LeadInfo(
            name=state["user_name"],
            email=state["user_email"],
            platform=state["user_platform"]
        )
        if not lead.is_complete() and not state["lead_captured"]:
            return "collect_lead_info"
        return "end"
    
    def is_lead_complete(self, state: AgentState) -> Literal["capture_lead", "end"]:
        """Check if all lead info has been collected"""
        last_message = state["messages"][-1]  # Last user message
        if isinstance(last_message, HumanMessage):
            state = self._extract_lead_info(state, last_message.content)
        
        lead = LeadInfo(
            name=state["user_name"],
            email=state["user_email"],
            platform=state["user_platform"]
        )
        
        if lead.is_complete():
            return "capture_lead"
        return "end"
    
    def _extract_lead_info(self, state: AgentState, message: str) -> AgentState:
        """Extract lead information from user message using regex patterns"""
        message_lower = message.lower()
        
        # Extract name (look for "my name is" or "i'm" or simple capitalized words)
        name_patterns = [
            r"my name is\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)?)",
            r"i'm\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)?)",
            r"this is\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)?)",
            r"call me\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)?)"
        ]
        for pattern in name_patterns:
            match = re.search(pattern, message)
            if match:
                state["user_name"] = match.group(1)
                break
        
        # Extract email
        email_pattern = r"([a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,})"
        email_match = re.search(email_pattern, message)
        if email_match:
            state["user_email"] = email_match.group(1)
        
        # Extract platform
        platforms = ["youtube", "instagram", "tiktok", "twitter", "linkedin", "facebook"]
        for platform in platforms:
            if platform in message_lower:
                state["user_platform"] = platform.capitalize()
                break
        
        return state
    
    def _build_system_prompt(self, state: AgentState) -> str:
        """Build dynamic system prompt based on state"""
        base_prompt = """You are a friendly and knowledgeable AutoStream sales assistant. 
AutoStream is an AI-powered video editing platform for content creators.

Your responsibilities:
1. Answer questions about AutoStream pricing, features, and policies accurately
2. Help users understand which plan (Basic or Pro) fits their needs
3. Identify when a user is showing high intent to purchase
4. When detecting high-intent, politely collect their name, email, and content platform

Current Context:
"""
        if state["knowledge_context"]:
            base_prompt += f"\nRelevant Knowledge:\n{state['knowledge_context']}\n"
        
        base_prompt += f"\nUser Intent: {state['intent']}\n"
        
        if state["user_name"]:
            base_prompt += f"User Name: {state['user_name']}\n"
        if state["user_email"]:
            base_prompt += f"User Email: {state['user_email']}\n"
        if state["user_platform"]:
            base_prompt += f"User Platform: {state['user_platform']}\n"
        
        base_prompt += """
Guidelines:
- Be conversational and enthusiastic
- Reference specific pricing and features from the knowledge base
- For high-intent users, gently ask for missing information (name, email, platform)
- Only mention that you're capturing a lead after ALL information is collected
- Be helpful and address concerns
- If unsure about something, ask the user or admit the limitation
"""
        return base_prompt
    
    def chat(self, user_message: str) -> str:
        """Process a user message and return agent response"""
        # Initialize state for first message
        state = {
            "messages": [],
            "intent": None,
            "user_name": None,
            "user_email": None,
            "user_platform": None,
            "lead_captured": False,
            "conversation_turn": 0,
            "knowledge_context": None
        }
        
        # Add user message
        state["messages"].append(HumanMessage(content=user_message))
        
        # Run the graph
        result = self.graph.invoke(state)
        
        # Extract final response
        if result["messages"]:
            last_message = result["messages"][-1]
            return last_message.content if isinstance(last_message, AIMessage) else str(last_message)
        
        return "I couldn't generate a response. Please try again."
    
    def multi_turn_chat(self, messages: List[str]) -> List[Dict[str, str]]:
        """
        Run a multi-turn conversation
        Returns list of {role, content} for the entire conversation
        """
        state = {
            "messages": [],
            "intent": None,
            "user_name": None,
            "user_email": None,
            "user_platform": None,
            "lead_captured": False,
            "conversation_turn": 0,
            "knowledge_context": None
        }
        
        conversation_history = []
        
        for user_message in messages:
            # Add user message
            state["messages"].append(HumanMessage(content=user_message))
            conversation_history.append({
                "role": "user",
                "content": user_message
            })
            
            # Run the graph
            result = self.graph.invoke(state)
            
            # Update state for next iteration
            state = result
            
            # Extract and store agent response
            if result["messages"]:
                last_message = result["messages"][-1]
                if isinstance(last_message, AIMessage):
                    conversation_history.append({
                        "role": "assistant",
                        "content": last_message.content
                    })
        
        return conversation_history


# ============================================================================
# Main Entry Point
# ============================================================================

if __name__ == "__main__":
    # Initialize agent
    agent = AutoStreamAgent()
    
    # Example multi-turn conversation
    print("🤖 AutoStream Sales Agent Started\n")
    print("="*60)
    
    sample_conversation = [
        "Hi, I'm interested in video editing software",
        "Can you tell me about your pricing?",
        "That sounds great! I'd like to try the Pro plan for my YouTube channel",
        "My name is Sarah Johnson and my email is sarah.johnson@email.com"
    ]
    
    conversation = agent.multi_turn_chat(sample_conversation)
    
    print("\n📝 Conversation Summary:")
    print("="*60)
    for turn in conversation:
        role = "👤 User" if turn["role"] == "user" else "🤖 Agent"
        print(f"\n{role}:")
        print(f"{turn['content']}")
    
    print("\n" + "="*60)
    print("✅ Demo conversation completed!")
