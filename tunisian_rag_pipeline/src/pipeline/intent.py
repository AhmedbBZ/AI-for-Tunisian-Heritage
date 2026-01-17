"""
Intent Classification Module
============================
Classifies user queries to route them appropriately.
"""

from typing import Optional, List, Tuple
from dataclasses import dataclass
from enum import Enum
import re

from loguru import logger


class QueryIntent(Enum):
    """Types of user query intents."""
    FACTUAL_QUESTION = "factual_question"
    HISTORICAL_EVENT = "historical_event"
    PERSON_INFO = "person_info"
    CULTURAL_TOPIC = "cultural_topic"
    COMPARISON = "comparison"
    SUMMARY_REQUEST = "summary_request"
    DEFINITION = "definition"
    TIMELINE = "timeline"
    GENERAL_CHAT = "general_chat"


@dataclass
class IntentResult:
    """Result of intent classification."""
    intent: QueryIntent
    confidence: float
    language: str
    entities: List[str]
    keywords: List[str]


class IntentClassifier:
    """
    Classifies user queries into intents for better routing and response.
    
    Uses rule-based classification with keyword matching.
    Can be enhanced with ML-based classification if needed.
    """
    
    # Intent patterns (multilingual)
    INTENT_PATTERNS = {
        QueryIntent.FACTUAL_QUESTION: {
            "en": [r"^(what|who|when|where|which|how many)\b", r"\?$"],
            "fr": [r"^(qu'est|qui|quand|où|quel|combien)\b", r"\?$"],
            "ar": [r"(ما|من|متى|أين|كم|كيف)", r"؟$"]
        },
        QueryIntent.HISTORICAL_EVENT: {
            "en": [r"\b(revolution|war|independence|battle|uprising|protest)\b", r"\b(happened|occurred|took place)\b"],
            "fr": [r"\b(révolution|guerre|indépendance|bataille|soulèvement)\b"],
            "ar": [r"(ثورة|حرب|استقلال|معركة|انتفاضة)"]
        },
        QueryIntent.PERSON_INFO: {
            "en": [r"\b(who was|who is|tell me about)\b", r"\b(president|leader|founder|hero)\b"],
            "fr": [r"\b(qui était|qui est|parlez-moi de)\b"],
            "ar": [r"(من كان|من هو|أخبرني عن)"]
        },
        QueryIntent.CULTURAL_TOPIC: {
            "en": [r"\b(culture|tradition|custom|heritage|art|music|food|cuisine)\b"],
            "fr": [r"\b(culture|tradition|coutume|patrimoine|art|musique|cuisine)\b"],
            "ar": [r"(ثقافة|تقاليد|تراث|فن|موسيقى|طعام)"]
        },
        QueryIntent.COMPARISON: {
            "en": [r"\b(compare|difference|versus|vs|between)\b", r"\b(similar|different)\b"],
            "fr": [r"\b(comparer|différence|entre|similaire|différent)\b"],
            "ar": [r"(قارن|الفرق|بين|مشابه|مختلف)"]
        },
        QueryIntent.SUMMARY_REQUEST: {
            "en": [r"\b(summarize|summary|overview|brief|explain)\b", r"^tell me about\b"],
            "fr": [r"\b(résumer|résumé|aperçu|bref|expliquer)\b"],
            "ar": [r"(لخص|ملخص|نظرة عامة|شرح)"]
        },
        QueryIntent.DEFINITION: {
            "en": [r"\b(what is|define|definition|meaning of)\b"],
            "fr": [r"\b(qu'est-ce que|définir|définition|signification)\b"],
            "ar": [r"(ما هو|عرف|تعريف|معنى)"]
        },
        QueryIntent.TIMELINE: {
            "en": [r"\b(timeline|chronology|sequence|order of events)\b", r"\b(first|then|after|before)\b.*\b(happen|occur)\b"],
            "fr": [r"\b(chronologie|séquence|ordre des événements)\b"],
            "ar": [r"(تسلسل زمني|ترتيب الأحداث)"]
        }
    }
    
    # Keywords for entity extraction
    ENTITY_KEYWORDS = {
        "people": [
            "Bourguiba", "Ben Ali", "Bouazizi", "Hannibal",
            "بورقيبة", "بن علي", "البوعزيزي"
        ],
        "places": [
            "Tunisia", "Tunis", "Carthage", "Kairouan", "Sousse", "Sfax", "Bizerte",
            "تونس", "قرطاج", "القيروان", "سوسة", "صفاقس", "بنزرت"
        ],
        "events": [
            "revolution", "independence", "protectorate", "Arab Spring",
            "ثورة", "استقلال"
        ],
        "periods": [
            "ancient", "medieval", "colonial", "modern", "French",
            "قديم", "استعماري", "حديث"
        ]
    }
    
    def __init__(self):
        """Initialize the intent classifier."""
        logger.info("Initialized IntentClassifier")
    
    def classify(self, query: str) -> IntentResult:
        """
        Classify a user query.
        
        Args:
            query: User's query text
            
        Returns:
            IntentResult with classification
        """
        # Detect language
        language = self._detect_language(query)
        
        # Extract entities and keywords
        entities = self._extract_entities(query)
        keywords = self._extract_keywords(query)
        
        # Classify intent
        intent, confidence = self._classify_intent(query, language)
        
        return IntentResult(
            intent=intent,
            confidence=confidence,
            language=language,
            entities=entities,
            keywords=keywords
        )
    
    def _detect_language(self, query: str) -> str:
        """Detect query language."""
        # Check for Arabic characters
        if re.search(r'[\u0600-\u06FF]', query):
            return "ar"
        
        # Check for French-specific patterns
        french_indicators = ['est-ce', 'qu\'', 'où', 'é', 'è', 'ê', 'ç', 'à']
        query_lower = query.lower()
        for indicator in french_indicators:
            if indicator in query_lower:
                return "fr"
        
        return "en"
    
    def _classify_intent(self, query: str, language: str) -> Tuple[QueryIntent, float]:
        """
        Classify intent based on patterns.
        
        Returns:
            Tuple of (intent, confidence)
        """
        query_lower = query.lower()
        
        scores = {}
        
        for intent, lang_patterns in self.INTENT_PATTERNS.items():
            patterns = lang_patterns.get(language, []) + lang_patterns.get("en", [])
            
            matches = 0
            for pattern in patterns:
                if re.search(pattern, query_lower, re.IGNORECASE):
                    matches += 1
            
            if patterns:
                scores[intent] = matches / len(patterns)
        
        # Find best match
        if scores:
            best_intent = max(scores, key=scores.get)
            confidence = scores[best_intent]
            
            if confidence > 0:
                return best_intent, min(confidence + 0.3, 1.0)  # Boost confidence
        
        # Default to factual question if has question mark
        if "?" in query or "؟" in query:
            return QueryIntent.FACTUAL_QUESTION, 0.5
        
        return QueryIntent.GENERAL_CHAT, 0.3
    
    def _extract_entities(self, query: str) -> List[str]:
        """Extract named entities from query."""
        entities = []
        
        for category, keywords in self.ENTITY_KEYWORDS.items():
            for keyword in keywords:
                if keyword.lower() in query.lower():
                    entities.append(keyword)
        
        return list(set(entities))
    
    def _extract_keywords(self, query: str) -> List[str]:
        """Extract important keywords from query."""
        # Remove common words
        stop_words = {
            "the", "a", "an", "is", "are", "was", "were", "what", "when", "where",
            "who", "how", "why", "can", "could", "would", "should", "do", "does",
            "le", "la", "les", "un", "une", "est", "sont", "que", "qui", "où",
            "في", "من", "إلى", "على", "هل", "ما", "كيف"
        }
        
        # Tokenize and filter
        words = re.findall(r'\b\w+\b', query.lower())
        keywords = [w for w in words if w not in stop_words and len(w) > 2]
        
        return keywords[:10]  # Limit to 10 keywords
    
    def get_search_suggestions(self, intent_result: IntentResult) -> List[str]:
        """
        Get search query suggestions based on intent.
        
        Args:
            intent_result: Classification result
            
        Returns:
            List of suggested search queries
        """
        suggestions = []
        
        # Add entities as search terms
        suggestions.extend(intent_result.entities)
        
        # Add keywords
        suggestions.extend(intent_result.keywords[:5])
        
        # Add intent-specific terms
        intent_terms = {
            QueryIntent.HISTORICAL_EVENT: ["history", "event", "date"],
            QueryIntent.PERSON_INFO: ["biography", "life", "achievements"],
            QueryIntent.CULTURAL_TOPIC: ["culture", "tradition", "heritage"],
            QueryIntent.TIMELINE: ["timeline", "chronology", "dates"],
        }
        
        if intent_result.intent in intent_terms:
            suggestions.extend(intent_terms[intent_result.intent])
        
        return list(set(suggestions))


def classify_query(query: str) -> IntentResult:
    """
    Convenience function to classify a query.
    
    Args:
        query: User's query
        
    Returns:
        IntentResult
    """
    classifier = IntentClassifier()
    return classifier.classify(query)


if __name__ == "__main__":
    # Test intent classification
    test_queries = [
        "When did the Tunisian revolution begin?",
        "Who was Habib Bourguiba?",
        "Tell me about Tunisian culture",
        "Compare ancient Carthage with modern Tunisia",
        "La révolution tunisienne de 2011",
        "متى بدأت الثورة التونسية؟",
        "What is the history of Kairouan?",
    ]
    
    classifier = IntentClassifier()
    
    print("Intent Classification Tests:")
    print("=" * 60)
    
    for query in test_queries:
        result = classifier.classify(query)
        print(f"\nQuery: {query}")
        print(f"  Intent: {result.intent.value}")
        print(f"  Confidence: {result.confidence:.2f}")
        print(f"  Language: {result.language}")
        print(f"  Entities: {result.entities}")
        print(f"  Keywords: {result.keywords[:5]}")
