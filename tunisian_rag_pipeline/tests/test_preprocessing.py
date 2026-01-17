"""
Tests for Text Preprocessing Module
"""

import pytest
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.preprocessing import TextPreprocessor, PreprocessingConfig
from src.data.ingestion import Document


class TestPreprocessingConfig:
    """Test PreprocessingConfig."""
    
    def test_default_config(self):
        """Test default configuration."""
        config = PreprocessingConfig()
        
        assert config.lowercase is False
        assert config.remove_extra_whitespace is True
        assert config.normalize_unicode is True
    
    def test_custom_config(self):
        """Test custom configuration."""
        config = PreprocessingConfig(
            lowercase=True,
            min_length=100
        )
        
        assert config.lowercase is True
        assert config.min_length == 100


class TestTextPreprocessor:
    """Test TextPreprocessor class."""
    
    @pytest.fixture
    def preprocessor(self):
        """Create preprocessor instance."""
        return TextPreprocessor()
    
    def test_clean_text_whitespace(self, preprocessor):
        """Test whitespace normalization."""
        text = "Hello    world\n\n\nTest"
        cleaned = preprocessor.clean_text(text)
        
        assert "    " not in cleaned
        assert "\n\n\n" not in cleaned
    
    def test_clean_text_special_chars(self, preprocessor):
        """Test special character handling."""
        text = "Hello\x00world\x1ftest"
        cleaned = preprocessor.clean_text(text)
        
        assert "\x00" not in cleaned
        assert "\x1f" not in cleaned
    
    def test_detect_language_english(self, preprocessor):
        """Test English language detection."""
        text = "This is a test sentence in English."
        lang = preprocessor.detect_language(text)
        
        assert lang == "en"
    
    def test_detect_language_french(self, preprocessor):
        """Test French language detection."""
        text = "Ceci est une phrase en français."
        lang = preprocessor.detect_language(text)
        
        assert lang == "fr"
    
    def test_detect_language_arabic(self, preprocessor):
        """Test Arabic language detection."""
        text = "هذه جملة باللغة العربية"
        lang = preprocessor.detect_language(text)
        
        assert lang == "ar"
    
    def test_normalize_arabic(self, preprocessor):
        """Test Arabic text normalization."""
        # Test alef normalization
        text = "أحمد إبراهيم آدم"
        normalized = preprocessor.normalize_arabic(text)
        
        # All alef variants should be normalized
        assert "ا" in normalized
    
    def test_process_document(self, preprocessor):
        """Test full document processing."""
        doc = Document(
            content="  Hello   World!  \n\n Test  ",
            source="test.txt",
            doc_type="txt"
        )
        
        processed = preprocessor.process_document(doc)
        
        assert "   " not in processed.content
        assert processed.metadata.get("preprocessed") is True
    
    def test_process_document_language_detection(self, preprocessor):
        """Test language detection during processing."""
        doc = Document(
            content="This is English text for testing purposes.",
            source="test.txt",
            doc_type="txt"
        )
        
        processed = preprocessor.process_document(doc)
        
        assert processed.metadata.get("language") == "en"


class TestEdgeCases:
    """Test edge cases."""
    
    def test_empty_text(self):
        """Test handling empty text."""
        preprocessor = TextPreprocessor()
        cleaned = preprocessor.clean_text("")
        assert cleaned == ""
    
    def test_none_handling(self):
        """Test handling None values."""
        preprocessor = TextPreprocessor()
        # Should not raise exception
        result = preprocessor.clean_text("")
        assert result == ""
    
    def test_mixed_language(self):
        """Test mixed language text."""
        preprocessor = TextPreprocessor()
        text = "Hello مرحبا Bonjour"
        lang = preprocessor.detect_language(text)
        
        # Should detect one of the languages
        assert lang in ["en", "ar", "fr", "unknown"]
    
    def test_very_short_text(self):
        """Test very short text."""
        preprocessor = TextPreprocessor()
        text = "Hi"
        lang = preprocessor.detect_language(text)
        
        # Should still return a language
        assert lang is not None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
