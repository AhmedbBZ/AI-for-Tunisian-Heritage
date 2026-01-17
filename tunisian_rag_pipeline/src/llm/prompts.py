"""
Prompt Templates Module
=======================
Contains prompt templates for different query types and languages.
"""

from typing import Dict, List, Any, Optional
from dataclasses import dataclass


@dataclass
class PromptTemplate:
    """A prompt template with placeholders."""
    template: str
    input_variables: List[str]
    
    def format(self, **kwargs) -> str:
        """Format the template with provided values."""
        return self.template.format(**kwargs)


# =============================================================================
# SYSTEM PROMPTS
# =============================================================================

SYSTEM_PROMPT_RAG = """You are a knowledgeable AI assistant specializing in Tunisian history, culture, and heritage. Your role is to provide accurate, informative answers based on the provided context.

Guidelines:
1. Base your answers PRIMARILY on the provided context/evidence
2. If the context doesn't contain enough information, say so clearly
3. Cite specific sources when possible
4. Be factual and avoid speculation
5. For historical events, mention dates and key figures when available
6. Respect cultural sensitivity when discussing heritage topics
7. You can provide brief additional context if it enhances understanding, but prioritize the evidence

Language: Respond in the same language as the user's question, unless asked otherwise."""


SYSTEM_PROMPT_MULTILINGUAL = """You are a multilingual AI assistant specializing in Tunisian history and heritage. You can respond in English, French, and Arabic.

Important:
- Answer in the same language as the question
- Base answers on the provided evidence
- If evidence is in a different language than the question, translate key points
- Cite sources appropriately"""


# =============================================================================
# RAG PROMPT TEMPLATES
# =============================================================================

RAG_PROMPT_TEMPLATE = PromptTemplate(
    template="""Based on the following context, please answer the question.

CONTEXT:
{context}

QUESTION: {question}

INSTRUCTIONS:
- Answer based on the context provided above
- If the context doesn't contain relevant information, say "Based on the available information, I cannot fully answer this question"
- Include specific details and cite sources when possible
- Be concise but comprehensive

ANSWER:""",
    input_variables=["context", "question"]
)


RAG_PROMPT_WITH_SOURCES = PromptTemplate(
    template="""You are answering questions about Tunisian heritage based on retrieved documents.

RETRIEVED EVIDENCE:
{context}

USER QUESTION: {question}

Please provide:
1. A clear, informative answer based on the evidence
2. Reference to which sources support your answer

Format your response as:
ANSWER: [Your answer here]

SOURCES: [List the relevant source references]

Begin:""",
    input_variables=["context", "question"]
)


RAG_PROMPT_DETAILED = PromptTemplate(
    template="""You are a Tunisian heritage expert assistant. Use the following evidence to answer the question.

EVIDENCE DOCUMENTS:
{context}

QUESTION: {question}

Provide a detailed answer that:
- Directly addresses the question
- Uses specific information from the evidence
- Acknowledges any limitations in the available information
- Is well-organized and easy to read

DETAILED ANSWER:""",
    input_variables=["context", "question"]
)


# =============================================================================
# INTENT CLASSIFICATION PROMPTS
# =============================================================================

INTENT_CLASSIFICATION_PROMPT = PromptTemplate(
    template="""Classify the following question into one of these categories:
- factual_question: Direct factual query (who, what, when, where)
- historical_event: Question about a historical event
- person_info: Question about a person
- cultural_topic: Question about culture, traditions, or customs
- comparison: Comparing different things or periods
- summary_request: Request for a summary or overview
- general_chat: Casual conversation or unclear intent

Question: {question}

Classification:""",
    input_variables=["question"]
)


# =============================================================================
# LANGUAGE-SPECIFIC PROMPTS
# =============================================================================

ARABIC_RAG_PROMPT = PromptTemplate(
    template="""أنت مساعد ذكي متخصص في التراث والتاريخ التونسي.

المعلومات المتاحة:
{context}

السؤال: {question}

التعليمات:
- أجب بناءً على المعلومات المقدمة أعلاه
- إذا لم تكن المعلومات كافية، اذكر ذلك بوضوح
- استخدم اللغة العربية الفصحى

الإجابة:""",
    input_variables=["context", "question"]
)


FRENCH_RAG_PROMPT = PromptTemplate(
    template="""Vous êtes un assistant spécialisé dans l'histoire et le patrimoine tunisien.

CONTEXTE:
{context}

QUESTION: {question}

INSTRUCTIONS:
- Répondez en vous basant sur le contexte fourni
- Si les informations sont insuffisantes, indiquez-le clairement
- Citez vos sources quand c'est possible

RÉPONSE:""",
    input_variables=["context", "question"]
)


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def get_rag_prompt(
    question: str,
    context: str,
    language: Optional[str] = None,
    detailed: bool = False,
    include_sources: bool = True
) -> str:
    """
    Get the appropriate RAG prompt based on parameters.
    
    Args:
        question: User's question
        context: Retrieved context
        language: Language code (ar, fr, en) or None for auto
        detailed: Whether to use detailed prompt
        include_sources: Whether to request source citations
        
    Returns:
        Formatted prompt string
    """
    # Auto-detect language from question
    if language is None:
        language = _detect_question_language(question)
    
    # Select prompt template
    if language == "ar":
        template = ARABIC_RAG_PROMPT
    elif language == "fr":
        template = FRENCH_RAG_PROMPT
    elif detailed:
        template = RAG_PROMPT_DETAILED
    elif include_sources:
        template = RAG_PROMPT_WITH_SOURCES
    else:
        template = RAG_PROMPT_TEMPLATE
    
    return template.format(context=context, question=question)


def _detect_question_language(question: str) -> str:
    """Simple language detection for questions."""
    import re
    
    # Check for Arabic characters
    if re.search(r'[\u0600-\u06FF]', question):
        return "ar"
    
    # Check for French-specific characters/words
    french_markers = ['qu', 'est-ce', 'comment', 'pourquoi', 'où', 'qui', 'quoi', 'é', 'è', 'ê', 'à', 'ç']
    question_lower = question.lower()
    for marker in french_markers:
        if marker in question_lower:
            return "fr"
    
    return "en"


def format_context_for_prompt(
    chunks: List[Dict[str, Any]],
    max_length: int = 3000,
    include_metadata: bool = True
) -> str:
    """
    Format retrieved chunks into context string for prompts.
    
    Args:
        chunks: List of chunk dictionaries
        max_length: Maximum context length
        include_metadata: Whether to include source metadata
        
    Returns:
        Formatted context string
    """
    context_parts = []
    current_length = 0
    
    for i, chunk in enumerate(chunks):
        content = chunk.get("content", "")
        
        # Build chunk text
        if include_metadata:
            source = chunk.get("source_info", chunk.get("metadata", {}).get("source_file", f"Source {i+1}"))
            chunk_text = f"[{source}]\n{content}"
        else:
            chunk_text = content
        
        # Check length
        if current_length + len(chunk_text) > max_length:
            # Truncate if needed
            remaining = max_length - current_length
            if remaining > 100:
                chunk_text = chunk_text[:remaining] + "..."
                context_parts.append(chunk_text)
            break
        
        context_parts.append(chunk_text)
        current_length += len(chunk_text) + 4  # +4 for separators
    
    return "\n\n---\n\n".join(context_parts)


def build_chat_messages(
    question: str,
    context: str,
    system_prompt: Optional[str] = None,
    chat_history: Optional[List[Dict[str, str]]] = None
) -> List[Dict[str, str]]:
    """
    Build chat messages for chat-based LLMs.
    
    Args:
        question: User's question
        context: Retrieved context
        system_prompt: System prompt to use
        chat_history: Previous chat messages
        
    Returns:
        List of message dictionaries
    """
    messages = []
    
    # System message
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    else:
        messages.append({"role": "system", "content": SYSTEM_PROMPT_RAG})
    
    # Chat history
    if chat_history:
        messages.extend(chat_history)
    
    # Current query with context
    user_message = get_rag_prompt(question, context)
    messages.append({"role": "user", "content": user_message})
    
    return messages


# =============================================================================
# PROMPT REGISTRY
# =============================================================================

PROMPT_REGISTRY = {
    "rag_basic": RAG_PROMPT_TEMPLATE,
    "rag_with_sources": RAG_PROMPT_WITH_SOURCES,
    "rag_detailed": RAG_PROMPT_DETAILED,
    "rag_arabic": ARABIC_RAG_PROMPT,
    "rag_french": FRENCH_RAG_PROMPT,
    "intent_classification": INTENT_CLASSIFICATION_PROMPT,
}


def get_prompt(name: str) -> PromptTemplate:
    """Get a prompt template by name."""
    if name not in PROMPT_REGISTRY:
        raise ValueError(f"Unknown prompt: {name}. Available: {list(PROMPT_REGISTRY.keys())}")
    return PROMPT_REGISTRY[name]


if __name__ == "__main__":
    # Test prompts
    print("Available prompts:", list(PROMPT_REGISTRY.keys()))
    
    # Test RAG prompt
    test_context = """
    [wikipedia_Tunisian_revolution.txt]
    The Tunisian Revolution began in December 2010 after Mohamed Bouazizi's self-immolation.
    
    [sample_resistance_sousse.txt]
    في عام ١٩٥٢، قاد محمد الزواري مجموعة من المقاومين في مدينة سوسة.
    """
    
    test_question = "When did the Tunisian revolution begin?"
    
    prompt = get_rag_prompt(test_question, test_context)
    print("\n" + "="*50)
    print("Generated Prompt:")
    print("="*50)
    print(prompt[:500] + "...")
