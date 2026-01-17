"""
Tests for Pipeline Module
"""

import pytest
import tempfile
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.pipeline.intent import IntentClassifier, QueryIntent


class TestQueryIntent:
    """Test QueryIntent enum."""
    
    def test_intent_values(self):
        """Test intent enum values."""
        assert QueryIntent.FACTUAL.value == "factual"
        assert QueryIntent.EXPLORATORY.value == "exploratory"
        assert QueryIntent.COMPARATIVE.value == "comparative"


class TestIntentClassifier:
    """Test IntentClassifier class."""
    
    @pytest.fixture
    def classifier(self):
        """Create classifier instance."""
        return IntentClassifier()
    
    def test_classify_factual_english(self, classifier):
        """Test classifying factual question in English."""
        result = classifier.classify("What is the capital of Tunisia?")
        
        assert result.intent == QueryIntent.FACTUAL
        assert result.language == "en"
        assert 0 <= result.confidence <= 1
    
    def test_classify_factual_french(self, classifier):
        """Test classifying factual question in French."""
        result = classifier.classify("Quelle est la capitale de la Tunisie?")
        
        assert result.intent == QueryIntent.FACTUAL
        assert result.language == "fr"
    
    def test_classify_factual_arabic(self, classifier):
        """Test classifying factual question in Arabic."""
        result = classifier.classify("ما هي عاصمة تونس؟")
        
        assert result.intent == QueryIntent.FACTUAL
        assert result.language == "ar"
    
    def test_classify_exploratory(self, classifier):
        """Test classifying exploratory question."""
        result = classifier.classify("Tell me about Tunisian history")
        
        assert result.intent == QueryIntent.EXPLORATORY
    
    def test_classify_comparative(self, classifier):
        """Test classifying comparative question."""
        result = classifier.classify("Compare Tunisian and Moroccan cuisine")
        
        assert result.intent == QueryIntent.COMPARATIVE
    
    def test_classify_procedural(self, classifier):
        """Test classifying procedural question."""
        result = classifier.classify("How do I make couscous?")
        
        assert result.intent == QueryIntent.PROCEDURAL
    
    def test_classify_causal(self, classifier):
        """Test classifying causal question."""
        result = classifier.classify("Why did the revolution happen?")
        
        assert result.intent == QueryIntent.CAUSAL
    
    def test_confidence_range(self, classifier):
        """Test that confidence is in valid range."""
        queries = [
            "What is Tunisia?",
            "Tell me about the history",
            "How does this work?"
        ]
        
        for query in queries:
            result = classifier.classify(query)
            assert 0 <= result.confidence <= 1
    
    def test_entity_extraction(self, classifier):
        """Test entity extraction."""
        result = classifier.classify("What happened in Tunisia in 2011?")
        
        # Should extract some entities
        assert "entities" in result.metadata


class TestLanguageDetection:
    """Test language detection in intent classifier."""
    
    @pytest.fixture
    def classifier(self):
        return IntentClassifier()
    
    def test_english_detection(self, classifier):
        """Test English language detection."""
        result = classifier.classify("Hello, how are you?")
        assert result.language == "en"
    
    def test_french_detection(self, classifier):
        """Test French language detection."""
        result = classifier.classify("Bonjour, comment allez-vous?")
        assert result.language == "fr"
    
    def test_arabic_detection(self, classifier):
        """Test Arabic language detection."""
        result = classifier.classify("مرحبا، كيف حالك؟")
        assert result.language == "ar"
    
    def test_mixed_language(self, classifier):
        """Test mixed language handling."""
        result = classifier.classify("Hello مرحبا Bonjour")
        # Should detect primary language
        assert result.language in ["en", "fr", "ar"]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
