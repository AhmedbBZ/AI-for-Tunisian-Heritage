#!/usr/bin/env python3
"""
Fine-Tuning Script
==================
Script for fine-tuning and adapting models for the Tunisian Heritage RAG.

This script supports:
1. Embedding model fine-tuning with contrastive learning
2. LLM fine-tuning with LoRA/QLoRA adapters
3. Training data preparation from Q&A pairs

Usage:
    python fine_tune.py prepare-data --input qa_pairs.json
    python fine_tune.py train-embeddings --data training_data.json
    python fine_tune.py train-lora --model microsoft/Phi-3-mini-4k-instruct
"""

import os
import sys
import argparse
import json
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
import random

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from loguru import logger


@dataclass
class TrainingExample:
    """A training example for fine-tuning."""
    query: str
    positive: str  # Relevant passage
    negative: Optional[str] = None  # Hard negative
    answer: Optional[str] = None  # For instruction tuning


def load_qa_pairs(filepath: str) -> List[Dict[str, Any]]:
    """Load Q&A pairs from JSON file."""
    with open(filepath, 'r', encoding='utf-8') as f:
        return json.load(f)


def prepare_contrastive_data(
    documents: List[Dict[str, str]],
    output_path: str,
    num_negatives: int = 3
) -> None:
    """
    Prepare contrastive learning data for embedding fine-tuning.
    
    Creates query-positive-negative triplets where:
    - Query: A question or search query
    - Positive: Relevant passage
    - Negative: Non-relevant passage (hard negative)
    """
    logger.info("Preparing contrastive training data...")
    
    training_data = []
    
    # For each document, create query-passage pairs
    for i, doc in enumerate(documents):
        content = doc.get('content', '')
        source = doc.get('source', f'doc_{i}')
        
        # Skip if too short
        if len(content) < 100:
            continue
        
        # Create pseudo-queries from content
        # In real scenarios, you'd use actual user queries
        sentences = content.split('.')[:5]
        
        for sent in sentences:
            sent = sent.strip()
            if len(sent) < 20:
                continue
            
            # Use sentence as pseudo-query, full content as positive
            query = sent
            positive = content[:500]  # First 500 chars as positive
            
            # Sample random negatives from other documents
            negative_indices = random.sample(
                [j for j in range(len(documents)) if j != i],
                min(num_negatives, len(documents) - 1)
            )
            
            for neg_idx in negative_indices:
                negative = documents[neg_idx].get('content', '')[:500]
                
                training_data.append({
                    'query': query,
                    'positive': positive,
                    'negative': negative
                })
    
    # Save training data
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(training_data, f, indent=2, ensure_ascii=False)
    
    logger.info(f"Created {len(training_data)} training examples")
    logger.info(f"Saved to: {output_path}")


def prepare_instruction_data(
    qa_pairs: List[Dict[str, Any]],
    output_path: str,
    system_prompt: str = "You are a helpful assistant specializing in Tunisian history and culture."
) -> None:
    """
    Prepare instruction-following data for LLM fine-tuning.
    
    Creates examples in the format:
    - System prompt
    - User query + context
    - Expected answer
    """
    logger.info("Preparing instruction-following training data...")
    
    training_data = []
    
    for item in qa_pairs:
        question = item.get('question', item.get('query', ''))
        answer = item.get('answer', item.get('response', ''))
        context = item.get('context', '')
        
        if not question or not answer:
            continue
        
        # Create instruction format
        if context:
            instruction = f"""Based on the following context, answer the question.

Context: {context}

Question: {question}"""
        else:
            instruction = question
        
        training_data.append({
            'system': system_prompt,
            'instruction': instruction,
            'output': answer
        })
    
    # Save training data
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(training_data, f, indent=2, ensure_ascii=False)
    
    logger.info(f"Created {len(training_data)} instruction examples")
    logger.info(f"Saved to: {output_path}")


def train_embedding_model(
    training_data_path: str,
    base_model: str = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
    output_dir: str = "./fine_tuned_embedder",
    epochs: int = 3,
    batch_size: int = 16,
    learning_rate: float = 2e-5
) -> None:
    """
    Fine-tune embedding model using contrastive learning.
    
    Uses MultipleNegativesRankingLoss from sentence-transformers.
    """
    logger.info(f"Fine-tuning embedding model: {base_model}")
    
    try:
        from sentence_transformers import SentenceTransformer, InputExample, losses
        from torch.utils.data import DataLoader
    except ImportError:
        logger.error("Please install sentence-transformers: pip install sentence-transformers")
        return
    
    # Load training data
    with open(training_data_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    logger.info(f"Loaded {len(data)} training examples")
    
    # Create input examples
    train_examples = []
    for item in data:
        train_examples.append(InputExample(
            texts=[item['query'], item['positive'], item['negative']]
        ))
    
    # Initialize model
    model = SentenceTransformer(base_model)
    
    # Create dataloader
    train_dataloader = DataLoader(train_examples, shuffle=True, batch_size=batch_size)
    
    # Use contrastive loss
    train_loss = losses.MultipleNegativesRankingLoss(model)
    
    # Train
    logger.info("Starting training...")
    model.fit(
        train_objectives=[(train_dataloader, train_loss)],
        epochs=epochs,
        warmup_steps=100,
        output_path=output_dir,
        show_progress_bar=True
    )
    
    logger.info(f"Model saved to: {output_dir}")


def train_lora_adapter(
    training_data_path: str,
    base_model: str = "microsoft/Phi-3-mini-4k-instruct",
    output_dir: str = "./lora_adapter",
    epochs: int = 3,
    batch_size: int = 4,
    learning_rate: float = 2e-4,
    lora_r: int = 8,
    lora_alpha: int = 16,
    use_4bit: bool = True
) -> None:
    """
    Fine-tune LLM using LoRA/QLoRA.
    
    Creates a lightweight adapter that can be merged with base model.
    """
    logger.info(f"Fine-tuning LLM with LoRA: {base_model}")
    
    try:
        import torch
        from transformers import (
            AutoModelForCausalLM,
            AutoTokenizer,
            TrainingArguments,
            BitsAndBytesConfig
        )
        from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
        from trl import SFTTrainer
    except ImportError as e:
        logger.error(f"Missing dependency: {e}")
        logger.error("Install: pip install transformers peft trl bitsandbytes")
        return
    
    # Load training data
    with open(training_data_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    logger.info(f"Loaded {len(data)} training examples")
    
    # Format data for training
    def format_prompt(example):
        system = example.get('system', '')
        instruction = example['instruction']
        output = example['output']
        
        return f"""<|system|>{system}<|end|>
<|user|>{instruction}<|end|>
<|assistant|>{output}<|end|>"""
    
    formatted_data = [format_prompt(ex) for ex in data]
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    
    # Quantization config for QLoRA
    if use_4bit:
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16
        )
    else:
        bnb_config = None
    
    # Load model
    model = AutoModelForCausalLM.from_pretrained(
        base_model,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True
    )
    
    if use_4bit:
        model = prepare_model_for_kbit_training(model)
    
    # LoRA config
    lora_config = LoraConfig(
        r=lora_r,
        lora_alpha=lora_alpha,
        lora_dropout=0.1,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        bias="none",
        task_type="CAUSAL_LM"
    )
    
    # Apply LoRA
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    
    # Training arguments
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        gradient_accumulation_steps=4,
        learning_rate=learning_rate,
        fp16=True,
        logging_steps=10,
        save_steps=100,
        warmup_ratio=0.1,
        optim="paged_adamw_8bit" if use_4bit else "adamw_torch"
    )
    
    # Create dataset
    from datasets import Dataset
    dataset = Dataset.from_dict({"text": formatted_data})
    
    # Initialize trainer
    trainer = SFTTrainer(
        model=model,
        train_dataset=dataset,
        tokenizer=tokenizer,
        args=training_args,
        dataset_text_field="text",
        max_seq_length=2048
    )
    
    # Train
    logger.info("Starting LoRA training...")
    trainer.train()
    
    # Save adapter
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    
    logger.info(f"LoRA adapter saved to: {output_dir}")


def create_sample_qa_data(output_path: str, num_samples: int = 50) -> None:
    """Create sample Q&A data for testing fine-tuning."""
    logger.info("Creating sample Q&A data...")
    
    # Sample Tunisian heritage Q&A pairs
    samples = [
        {
            "question": "What is Tunisian couscous?",
            "answer": "Tunisian couscous is a traditional North African dish made from steamed semolina granules, typically served with a spiced vegetable or meat stew. It's a staple of Tunisian cuisine.",
            "context": "Couscous is the national dish of Tunisia..."
        },
        {
            "question": "What caused the Tunisian revolution?",
            "answer": "The Tunisian revolution of 2010-2011 was sparked by the self-immolation of Mohamed Bouazizi, protesting corruption and economic hardship. It led to the overthrow of President Ben Ali.",
            "context": "The Tunisian revolution began in December 2010..."
        },
        {
            "question": "What is the Medina of Tunis?",
            "answer": "The Medina of Tunis is a UNESCO World Heritage Site, containing over 700 historical monuments including palaces, mosques, and traditional markets dating back to the 7th century.",
            "context": "The Medina of Tunis is one of the oldest in the Arab world..."
        },
        {
            "question": "Who was Hannibal Barca?",
            "answer": "Hannibal Barca was a Carthaginian general from ancient Tunisia, famous for crossing the Alps with elephants to attack Rome during the Second Punic War (218-201 BC).",
            "context": "Hannibal was born in Carthage, located in modern-day Tunisia..."
        },
        {
            "question": "What is Bardo National Museum?",
            "answer": "The Bardo National Museum in Tunis houses one of the world's largest collections of Roman mosaics, alongside Carthaginian, Greek, and Islamic artifacts.",
            "context": "The Bardo Museum is located in a 15th-century palace..."
        }
    ]
    
    # Duplicate and vary samples to reach num_samples
    qa_data = []
    for i in range(num_samples):
        sample = samples[i % len(samples)].copy()
        qa_data.append(sample)
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(qa_data, f, indent=2, ensure_ascii=False)
    
    logger.info(f"Created {len(qa_data)} sample Q&A pairs")
    logger.info(f"Saved to: {output_path}")


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Fine-tuning scripts for Tunisian Heritage RAG"
    )
    
    subparsers = parser.add_subparsers(dest="command", help="Sub-commands")
    
    # Prepare data command
    prep_parser = subparsers.add_parser("prepare-data", help="Prepare training data")
    prep_parser.add_argument("--input", type=str, help="Input Q&A pairs JSON file")
    prep_parser.add_argument("--output", type=str, default="./training_data.json")
    prep_parser.add_argument("--type", choices=["contrastive", "instruction"], default="instruction")
    prep_parser.add_argument("--create-sample", action="store_true", help="Create sample data")
    
    # Train embeddings command
    embed_parser = subparsers.add_parser("train-embeddings", help="Fine-tune embedding model")
    embed_parser.add_argument("--data", type=str, required=True, help="Training data JSON")
    embed_parser.add_argument("--model", type=str, 
                             default="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
    embed_parser.add_argument("--output", type=str, default="./fine_tuned_embedder")
    embed_parser.add_argument("--epochs", type=int, default=3)
    embed_parser.add_argument("--batch-size", type=int, default=16)
    
    # Train LoRA command
    lora_parser = subparsers.add_parser("train-lora", help="Fine-tune LLM with LoRA")
    lora_parser.add_argument("--data", type=str, required=True, help="Training data JSON")
    lora_parser.add_argument("--model", type=str, default="microsoft/Phi-3-mini-4k-instruct")
    lora_parser.add_argument("--output", type=str, default="./lora_adapter")
    lora_parser.add_argument("--epochs", type=int, default=3)
    lora_parser.add_argument("--batch-size", type=int, default=4)
    lora_parser.add_argument("--no-4bit", action="store_true", help="Disable 4-bit quantization")
    
    return parser.parse_args()


def main():
    """Main function."""
    args = parse_args()
    
    # Setup logging
    logger.remove()
    logger.add(sys.stderr, level="INFO")
    
    if args.command == "prepare-data":
        if args.create_sample:
            create_sample_qa_data(args.output)
        elif args.input:
            qa_pairs = load_qa_pairs(args.input)
            if args.type == "instruction":
                prepare_instruction_data(qa_pairs, args.output)
            else:
                prepare_contrastive_data(qa_pairs, args.output)
        else:
            logger.error("Specify --input or --create-sample")
            sys.exit(1)
    
    elif args.command == "train-embeddings":
        train_embedding_model(
            training_data_path=args.data,
            base_model=args.model,
            output_dir=args.output,
            epochs=args.epochs,
            batch_size=args.batch_size
        )
    
    elif args.command == "train-lora":
        train_lora_adapter(
            training_data_path=args.data,
            base_model=args.model,
            output_dir=args.output,
            epochs=args.epochs,
            batch_size=args.batch_size,
            use_4bit=not args.no_4bit
        )
    
    else:
        logger.error("Specify a command: prepare-data, train-embeddings, or train-lora")
        logger.info("Use --help for more information")


if __name__ == "__main__":
    main()
