"""
Text Preprocessing Module
=========================
Handles text cleaning, normalization, and language detection.
Prepares documents for chunking and embedding.
"""

import re
import unicodedata
from typing import Optional, List, Tuple
from dataclasses import dataclass

from loguru import logger

# Optional imports with fallbacks
try:
    from langdetect import detect, detect_langs, LangDetectException
    LANGDETECT_AVAILABLE = True
except ImportError:
    LANGDETECT_AVAILABLE = False
    logger.warning("langdetect not installed. Language detection disabled.")


@dataclass
class PreprocessingConfig:
    """Configuration for text preprocessing."""
    min_text_length: int = 50
    max_text_length: int = 100000
    remove_extra_whitespace: bool = True
    normalize_unicode: bool = True
    remove_urls: bool = True
    remove_emails: bool = True
    remove_special_chars: bool = False
    lowercase: bool = False
    detect_language: bool = True


class TextPreprocessor:
    """
    Preprocesses text for the RAG pipeline.
    
    Features:
    - Unicode normalization
    - Whitespace cleanup
    - Language detection (Arabic, French, English)
    - URL/email removal
    - Character normalization
    """
    
    # Common patterns
    URL_PATTERN = re.compile(r'https?://\S+|www\.\S+')
    EMAIL_PATTERN = re.compile(r'\S+@\S+\.\S+')
    MULTIPLE_SPACES = re.compile(r' +')
    MULTIPLE_NEWLINES = re.compile(r'\n{3,}')
    
    # Arabic-specific patterns
    ARABIC_DIACRITICS = re.compile(r'[\u064B-\u065F\u0670]')
    
    def __init__(self, config: Optional[PreprocessingConfig] = None):
        """
        Initialize the preprocessor.
        
        Args:
            config: Preprocessing configuration
        """
        self.config = config or PreprocessingConfig()
        logger.info("Initialized TextPreprocessor")
    
    def clean_text(self, text: str) -> str:
        """
        Main text cleaning function.
        
        Args:
            text: Raw text to clean
            
        Returns:
            Cleaned text
        """
        if not text:
            return ""
        
        # Unicode normalization
        if self.config.normalize_unicode:
            text = self._normalize_unicode(text)
        
        # Remove URLs
        if self.config.remove_urls:
            text = self.URL_PATTERN.sub('', text)
        
        # Remove emails
        if self.config.remove_emails:
            text = self.EMAIL_PATTERN.sub('', text)
        
        # Clean whitespace
        if self.config.remove_extra_whitespace:
            text = self._clean_whitespace(text)
        
        # Lowercase (optional, usually not for multilingual)
        if self.config.lowercase:
            text = text.lower()
        
        return text.strip()
    
    def _normalize_unicode(self, text: str) -> str:
        """Normalize Unicode characters."""
        # NFC normalization for consistent character representation
        text = unicodedata.normalize('NFC', text)
        
        # Replace common problematic characters
        replacements = {
            '\u200b': '',  # Zero-width space
            '\u200c': '',  # Zero-width non-joiner
            '\u200d': '',  # Zero-width joiner
            '\ufeff': '',  # BOM
            '\xa0': ' ',   # Non-breaking space
            '"': '"',
            '"': '"',
            ''': "'",
            ''': "'",
            '–': '-',
            '—': '-',
            '…': '...',
        }
        
        for old, new in replacements.items():
            text = text.replace(old, new)
        
        return text
    
    def _clean_whitespace(self, text: str) -> str:
        """Clean up whitespace."""
        # Replace tabs with spaces
        text = text.replace('\t', ' ')
        
        # Collapse multiple spaces
        text = self.MULTIPLE_SPACES.sub(' ', text)
        
        # Collapse multiple newlines (keep max 2)
        text = self.MULTIPLE_NEWLINES.sub('\n\n', text)
        
        # Strip whitespace from lines
        lines = [line.strip() for line in text.split('\n')]
        text = '\n'.join(lines)
        
        return text
    
    def detect_language(self, text: str) -> Optional[str]:
        """
        Detect the primary language of the text.
        
        Args:
            text: Text to analyze
            
        Returns:
            Language code (en, fr, ar, etc.) or None
        """
        if not LANGDETECT_AVAILABLE:
            return self._detect_language_heuristic(text)
        
        try:
            # Use sample of text for detection (more reliable)
            sample = text[:5000] if len(text) > 5000 else text
            
            # Filter out very short text
            if len(sample.split()) < 10:
                return self._detect_language_heuristic(text)
            
            lang = detect(sample)
            return lang
        
        except LangDetectException:
            return self._detect_language_heuristic(text)
        except Exception as e:
            logger.warning(f"Language detection error: {e}")
            return None
    
    def _detect_language_heuristic(self, text: str) -> Optional[str]:
        """
        Simple heuristic-based language detection.
        Useful when langdetect is not available.
        """
        # Count Arabic characters
        arabic_chars = len(re.findall(r'[\u0600-\u06FF]', text))
        
        # Count French-specific characters
        french_chars = len(re.findall(r'[àâäéèêëïîôùûüçœæ]', text.lower()))
        
        total_chars = len(text)
        
        if total_chars == 0:
            return None
        
        arabic_ratio = arabic_chars / total_chars
        
        if arabic_ratio > 0.3:
            return "ar"
        elif french_chars > 5:
            return "fr"
        else:
            return "en"
    
    def detect_languages_multi(self, text: str) -> List[Tuple[str, float]]:
        """
        Detect multiple languages and their probabilities.
        
        Args:
            text: Text to analyze
            
        Returns:
            List of (language_code, probability) tuples
        """
        if not LANGDETECT_AVAILABLE:
            lang = self._detect_language_heuristic(text)
            return [(lang, 1.0)] if lang else []
        
        try:
            sample = text[:5000] if len(text) > 5000 else text
            langs = detect_langs(sample)
            return [(lang.lang, lang.prob) for lang in langs]
        except Exception:
            return []
    
    def normalize_arabic(self, text: str) -> str:
        """
        Normalize Arabic text.
        
        - Removes diacritics (tashkeel)
        - Normalizes various forms of letters
        """
        # Remove diacritics
        text = self.ARABIC_DIACRITICS.sub('', text)
        
        # Normalize alef variants
        text = re.sub(r'[إأآا]', 'ا', text)
        
        # Normalize yaa variants
        text = re.sub(r'[يى]', 'ي', text)
        
        # Normalize taa marbuta
        text = text.replace('ة', 'ه')
        
        return text
    
    def process_document(self, text: str, detect_lang: bool = True) -> Tuple[str, Optional[str]]:
        """
        Process a complete document.
        
        Args:
            text: Document text
            detect_lang: Whether to detect language
            
        Returns:
            Tuple of (cleaned_text, language_code)
        """
        # Clean the text
        cleaned = self.clean_text(text)
        
        # Check length constraints
        if len(cleaned) < self.config.min_text_length:
            logger.warning(f"Text too short after cleaning: {len(cleaned)} chars")
            return cleaned, None
        
        if len(cleaned) > self.config.max_text_length:
            logger.warning(f"Text truncated from {len(cleaned)} to {self.config.max_text_length} chars")
            cleaned = cleaned[:self.config.max_text_length]
        
        # Detect language
        language = None
        if detect_lang and self.config.detect_language:
            language = self.detect_language(cleaned)
        
        return cleaned, language
    
    def is_valid_text(self, text: str) -> bool:
        """
        Check if text is valid for processing.
        
        Args:
            text: Text to validate
            
        Returns:
            True if valid
        """
        if not text:
            return False
        
        cleaned = self.clean_text(text)
        
        if len(cleaned) < self.config.min_text_length:
            return False
        
        # Check for meaningful content (not just special chars)
        alpha_ratio = sum(1 for c in cleaned if c.isalpha()) / len(cleaned)
        if alpha_ratio < 0.3:
            return False
        
        return True


def preprocess_text(text: str, **kwargs) -> Tuple[str, Optional[str]]:
    """
    Convenience function for quick text preprocessing.
    
    Args:
        text: Text to preprocess
        **kwargs: Config options
        
    Returns:
        Tuple of (cleaned_text, language)
    """
    config = PreprocessingConfig(**kwargs)
    preprocessor = TextPreprocessor(config)
    return preprocessor.process_document(text)


if __name__ == "__main__":
    # Test the module
    test_texts = [
        # English
        "The Tunisian Revolution was a popular uprising that began in December 2010 and led to the fall of President Zine El Abidine Ben Ali.",
        
        # French
        "La révolution tunisienne a été un soulèvement populaire qui a commencé en décembre 2010.",
        
        # Arabic
        "الثورة التونسية هي انتفاضة شعبية بدأت في ديسمبر 2010 وأدت إلى سقوط الرئيس زين العابدين بن علي.",
        
        # Mixed with noise
        "  Check out https://example.com for more info!!!   \n\n\n  email@test.com  ",
    ]
    
    preprocessor = TextPreprocessor()
    
    print("Testing TextPreprocessor:\n")
    
    for text in test_texts:
        cleaned, lang = preprocessor.process_document(text)
        print(f"Original: {text[:60]}...")
        print(f"Cleaned:  {cleaned[:60]}...")
        print(f"Language: {lang}")
        print("-" * 50)
