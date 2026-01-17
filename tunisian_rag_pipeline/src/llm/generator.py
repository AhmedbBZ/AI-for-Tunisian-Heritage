"""
LLM Generator Module
====================
Handles LLM-based answer generation for RAG pipeline.
Supports multiple LLM providers: HuggingFace, OpenAI, Ollama.
"""

import os
from typing import List, Dict, Any, Optional, Generator
from dataclasses import dataclass
from enum import Enum

from loguru import logger


class LLMProvider(Enum):
    """Supported LLM providers."""
    HUGGINGFACE = "huggingface"
    OPENAI = "openai"
    OLLAMA = "ollama"
    LMSTUDIO = "lmstudio"


@dataclass
class LLMConfig:
    """Configuration for LLM generation."""
    provider: LLMProvider = LLMProvider.LMSTUDIO
    model_name: str = "saka-14b-i1"
    max_new_tokens: int = 512
    temperature: float = 0.7
    top_p: float = 0.9
    do_sample: bool = True
    device: str = "auto"
    load_in_4bit: bool = True
    load_in_8bit: bool = False
    trust_remote_code: bool = True
    # OpenAI specific
    openai_api_key: Optional[str] = None
    # Ollama specific
    ollama_base_url: str = "http://localhost:11434"
    # LM Studio specific
    lmstudio_base_url: str = "http://localhost:1234/v1"


@dataclass
class GenerationResult:
    """Result from LLM generation."""
    answer: str
    model: str
    tokens_used: int
    finish_reason: str
    raw_response: Optional[Any] = None


class LLMGenerator:
    """
    LLM-based answer generator for RAG pipeline.
    
    Supports:
    - HuggingFace models (local, with quantization)
    - OpenAI API
    - Ollama (local models)
    """
    
    def __init__(self, config: Optional[LLMConfig] = None):
        """
        Initialize the LLM generator.
        
        Args:
            config: LLM configuration
        """
        self.config = config or LLMConfig()
        self.model = None
        self.tokenizer = None
        self.pipeline = None
        
        logger.info(f"Initialized LLMGenerator with provider: {self.config.provider.value}")
    
    def _load_huggingface_model(self) -> None:
        """Load HuggingFace model with optional quantization."""
        if self.pipeline is not None:
            return
        
        import torch
        from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline, BitsAndBytesConfig
        
        logger.info(f"Loading HuggingFace model: {self.config.model_name}")
        
        # Determine device
        if self.config.device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            device = self.config.device
        
        # Quantization config
        quantization_config = None
        if device == "cuda":
            if self.config.load_in_4bit:
                quantization_config = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_compute_dtype=torch.float16,
                    bnb_4bit_use_double_quant=True,
                    bnb_4bit_quant_type="nf4"
                )
            elif self.config.load_in_8bit:
                quantization_config = BitsAndBytesConfig(load_in_8bit=True)
        
        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.config.model_name,
            trust_remote_code=self.config.trust_remote_code
        )
        
        # Ensure pad token
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Load model
        model_kwargs = {
            "trust_remote_code": self.config.trust_remote_code,
            "device_map": "auto" if device == "cuda" else None,
        }
        
        if quantization_config:
            model_kwargs["quantization_config"] = quantization_config
        elif device == "cuda":
            model_kwargs["torch_dtype"] = torch.float16
        
        self.model = AutoModelForCausalLM.from_pretrained(
            self.config.model_name,
            **model_kwargs
        )
        
        # Create pipeline
        self.pipeline = pipeline(
            "text-generation",
            model=self.model,
            tokenizer=self.tokenizer,
            device_map="auto" if device == "cuda" else None,
        )
        
        logger.info(f"Model loaded on device: {device}")
    
    def _generate_huggingface(
        self,
        prompt: str,
        **kwargs
    ) -> GenerationResult:
        """Generate using HuggingFace model."""
        self._load_huggingface_model()
        
        # Merge config with kwargs
        gen_kwargs = {
            "max_new_tokens": kwargs.get("max_new_tokens", self.config.max_new_tokens),
            "temperature": kwargs.get("temperature", self.config.temperature),
            "top_p": kwargs.get("top_p", self.config.top_p),
            "do_sample": kwargs.get("do_sample", self.config.do_sample),
            "pad_token_id": self.tokenizer.pad_token_id,
            "return_full_text": False,
        }
        
        # Generate
        outputs = self.pipeline(prompt, **gen_kwargs)
        
        generated_text = outputs[0]["generated_text"]
        
        # Clean up response
        answer = generated_text.strip()
        
        return GenerationResult(
            answer=answer,
            model=self.config.model_name,
            tokens_used=len(self.tokenizer.encode(answer)),
            finish_reason="stop",
            raw_response=outputs
        )
    
    def _generate_openai(
        self,
        prompt: str,
        messages: Optional[List[Dict[str, str]]] = None,
        **kwargs
    ) -> GenerationResult:
        """Generate using OpenAI API."""
        from openai import OpenAI
        
        api_key = self.config.openai_api_key or os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OpenAI API key not provided")
        
        client = OpenAI(api_key=api_key)
        
        # Build messages
        if messages is None:
            messages = [{"role": "user", "content": prompt}]
        
        response = client.chat.completions.create(
            model=self.config.model_name,
            messages=messages,
            max_tokens=kwargs.get("max_new_tokens", self.config.max_new_tokens),
            temperature=kwargs.get("temperature", self.config.temperature),
            top_p=kwargs.get("top_p", self.config.top_p),
        )
        
        answer = response.choices[0].message.content
        
        return GenerationResult(
            answer=answer,
            model=self.config.model_name,
            tokens_used=response.usage.total_tokens if response.usage else 0,
            finish_reason=response.choices[0].finish_reason,
            raw_response=response
        )
    
    def _generate_lmstudio(
        self,
        prompt: str,
        messages: Optional[List[Dict[str, str]]] = None,
        system_prompt: Optional[str] = None,
        **kwargs
    ) -> GenerationResult:
        """Generate using LM Studio API (OpenAI compatible)."""
        import requests
        
        url = f"{self.config.lmstudio_base_url}/chat/completions"
        
        # Build messages
        if messages is None:
            messages = []
            if system_prompt:
                messages.append({"role": "system", "content": system_prompt})
            messages.append({"role": "user", "content": prompt})
        
        payload = {
            "model": self.config.model_name,
            "messages": messages,
            "max_tokens": kwargs.get("max_new_tokens", self.config.max_new_tokens),
            "temperature": kwargs.get("temperature", self.config.temperature),
            "top_p": kwargs.get("top_p", self.config.top_p),
            "stream": False
        }
        
        logger.debug(f"LM Studio request to {url}")
        
        response = requests.post(url, json=payload, timeout=120)
        response.raise_for_status()
        
        result = response.json()
        
        answer = result["choices"][0]["message"]["content"]
        
        return GenerationResult(
            answer=answer,
            model=self.config.model_name,
            tokens_used=result.get("usage", {}).get("total_tokens", 0),
            finish_reason=result["choices"][0].get("finish_reason", "stop"),
            raw_response=result
        )
    
    def _generate_ollama(
        self,
        prompt: str,
        **kwargs
    ) -> GenerationResult:
        """Generate using Ollama API."""
        import requests
        
        url = f"{self.config.ollama_base_url}/api/generate"
        
        payload = {
            "model": self.config.model_name,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": kwargs.get("temperature", self.config.temperature),
                "top_p": kwargs.get("top_p", self.config.top_p),
                "num_predict": kwargs.get("max_new_tokens", self.config.max_new_tokens),
            }
        }
        
        response = requests.post(url, json=payload)
        response.raise_for_status()
        
        result = response.json()
        
        return GenerationResult(
            answer=result.get("response", ""),
            model=self.config.model_name,
            tokens_used=result.get("eval_count", 0),
            finish_reason="stop",
            raw_response=result
        )
    
    def generate(
        self,
        prompt: str,
        messages: Optional[List[Dict[str, str]]] = None,
        system_prompt: Optional[str] = None,
        **kwargs
    ) -> GenerationResult:
        """
        Generate answer from prompt.
        
        Args:
            prompt: The prompt string
            messages: Optional chat messages (for OpenAI/LMStudio)
            system_prompt: Optional system prompt
            **kwargs: Override generation parameters
            
        Returns:
            GenerationResult object
        """
        try:
            if self.config.provider == LLMProvider.HUGGINGFACE:
                return self._generate_huggingface(prompt, **kwargs)
            elif self.config.provider == LLMProvider.OPENAI:
                return self._generate_openai(prompt, messages, **kwargs)
            elif self.config.provider == LLMProvider.OLLAMA:
                return self._generate_ollama(prompt, **kwargs)
            elif self.config.provider == LLMProvider.LMSTUDIO:
                return self._generate_lmstudio(prompt, messages, system_prompt, **kwargs)
            else:
                raise ValueError(f"Unknown provider: {self.config.provider}")
        
        except Exception as e:
            logger.error(f"Generation error: {e}")
            raise
    
    def generate_with_context(
        self,
        question: str,
        context: str,
        system_prompt: Optional[str] = None,
        **kwargs
    ) -> GenerationResult:
        """
        Generate answer with retrieved context.
        
        Args:
            question: User's question
            context: Retrieved context/evidence
            system_prompt: Optional system prompt
            **kwargs: Generation parameters
            
        Returns:
            GenerationResult object
        """
        from .prompts import get_rag_prompt, SYSTEM_PROMPT_RAG
        
        # Build prompt
        prompt = get_rag_prompt(question, context)
        
        # For chat models, use messages format
        if self.config.provider in [LLMProvider.OPENAI, LLMProvider.LMSTUDIO]:
            messages = [
                {"role": "system", "content": system_prompt or SYSTEM_PROMPT_RAG},
                {"role": "user", "content": prompt}
            ]
            return self.generate(prompt, messages=messages, system_prompt=system_prompt or SYSTEM_PROMPT_RAG, **kwargs)
        else:
            # For completion models, prepend system prompt
            if system_prompt:
                full_prompt = f"{system_prompt}\n\n{prompt}"
            else:
                full_prompt = prompt
            return self.generate(full_prompt, **kwargs)
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the model."""
        info = {
            "provider": self.config.provider.value,
            "model_name": self.config.model_name,
            "max_new_tokens": self.config.max_new_tokens,
            "temperature": self.config.temperature,
        }
        
        if self.config.provider == LLMProvider.HUGGINGFACE:
            info["quantization"] = "4bit" if self.config.load_in_4bit else ("8bit" if self.config.load_in_8bit else "none")
        
        return info


class SimpleLLMGenerator:
    """
    Simplified LLM generator that works without heavy dependencies.
    Uses basic prompting with fallback options.
    """
    
    def __init__(self):
        """Initialize simple generator."""
        self.provider = None
        self._check_available_providers()
    
    def _check_available_providers(self) -> None:
        """Check which providers are available."""
        # Check for OpenAI
        if os.getenv("OPENAI_API_KEY"):
            try:
                from openai import OpenAI
                self.provider = "openai"
                logger.info("Using OpenAI provider")
                return
            except ImportError:
                pass
        
        # Check for Ollama
        try:
            import requests
            response = requests.get("http://localhost:11434/api/tags", timeout=2)
            if response.ok:
                self.provider = "ollama"
                logger.info("Using Ollama provider")
                return
        except:
            pass
        
        # Fallback to template-based response
        self.provider = "template"
        logger.info("Using template-based responses (no LLM available)")
    
    def generate(self, question: str, context: str) -> str:
        """
        Generate an answer.
        
        Args:
            question: User's question
            context: Retrieved context
            
        Returns:
            Generated answer string
        """
        if self.provider == "openai":
            return self._generate_openai(question, context)
        elif self.provider == "ollama":
            return self._generate_ollama(question, context)
        else:
            return self._generate_template(question, context)
    
    def _generate_openai(self, question: str, context: str) -> str:
        """Generate using OpenAI."""
        from openai import OpenAI
        
        client = OpenAI()
        
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are a helpful assistant specializing in Tunisian history and heritage. Answer based on the provided context."},
                {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {question}"}
            ],
            max_tokens=500,
            temperature=0.7
        )
        
        return response.choices[0].message.content
    
    def _generate_ollama(self, question: str, context: str) -> str:
        """Generate using Ollama."""
        import requests
        
        prompt = f"""You are a helpful assistant specializing in Tunisian history and heritage.

Context:
{context}

Question: {question}

Answer based on the context provided:"""
        
        response = requests.post(
            "http://localhost:11434/api/generate",
            json={
                "model": "llama2",
                "prompt": prompt,
                "stream": False
            }
        )
        
        return response.json().get("response", "")
    
    def _generate_template(self, question: str, context: str) -> str:
        """Generate template-based response when no LLM is available."""
        # Extract key information from context
        lines = context.split('\n')
        relevant_lines = [l for l in lines if l.strip() and not l.startswith('[')][:5]
        
        response = f"""Based on the available information:

{chr(10).join('• ' + l.strip() for l in relevant_lines)}

Note: This is a summary of retrieved information. For more detailed analysis, configure an LLM provider (OpenAI API key or Ollama)."""
        
        return response


def create_generator(
    provider: str = "huggingface",
    model_name: Optional[str] = None,
    **kwargs
) -> LLMGenerator:
    """
    Factory function to create an LLM generator.
    
    Args:
        provider: Provider name (huggingface, openai, ollama)
        model_name: Model name
        **kwargs: Additional config options
        
    Returns:
        LLMGenerator instance
    """
    provider_enum = LLMProvider(provider.lower())
    
    # Default models by provider
    default_models = {
        LLMProvider.HUGGINGFACE: "microsoft/Phi-3-mini-4k-instruct",
        LLMProvider.OPENAI: "gpt-3.5-turbo",
        LLMProvider.OLLAMA: "llama2",
    }
    
    if model_name is None:
        model_name = default_models.get(provider_enum, default_models[LLMProvider.HUGGINGFACE])
    
    config = LLMConfig(
        provider=provider_enum,
        model_name=model_name,
        **kwargs
    )
    
    return LLMGenerator(config)


if __name__ == "__main__":
    print("LLM Generator Module")
    print("=" * 50)
    
    # Test simple generator
    print("\nTesting SimpleLLMGenerator:")
    
    simple_gen = SimpleLLMGenerator()
    print(f"Provider: {simple_gen.provider}")
    
    test_context = """
    The Tunisian Revolution began in December 2010 following Mohamed Bouazizi's 
    self-immolation in Sidi Bouzid. The protests spread across the country, leading 
    to President Ben Ali's departure on January 14, 2011.
    """
    
    test_question = "When did the Tunisian revolution begin?"
    
    print(f"\nQuestion: {test_question}")
    answer = simple_gen.generate(test_question, test_context)
    print(f"\nAnswer:\n{answer}")
