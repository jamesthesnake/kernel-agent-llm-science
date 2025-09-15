#!/usr/bin/env python3
"""
KernelBench Model Trainer - Implement actual training loop for specialized CUDA kernel generation
"""

import os
import json
import torch
import logging
from pathlib import Path
from typing import Dict, List, Optional
from dataclasses import dataclass
from datetime import datetime

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class ModelTrainingConfig:
    """Configuration for model training"""
    model_name: str = "codellama/CodeLlama-7b-Instruct-hf"  # Use CodeLlama for CUDA kernel generation
    max_length: int = 4096  # Increased for longer CUDA code
    learning_rate: float = 2e-5  # Lower learning rate for larger model
    batch_size: int = 1  # Smaller batch for 7B model
    num_epochs: int = 2  # Fewer epochs to start
    save_steps: int = 50
    eval_steps: int = 25
    warmup_steps: int = 20
    output_dir: str = "kernelbench_codellama_model"
    logging_steps: int = 5


class KernelBenchDataset:
    """Dataset class for KernelBench training data"""

    def __init__(self, examples: List[Dict], tokenizer, max_length: int = 2048):
        self.examples = examples
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        example = self.examples[idx]

        # Create prompt and target using CodeLlama's instruction format
        prompt = f"""<s>[INST] You are an expert CUDA kernel developer. Generate an optimized CUDA kernel solution for this PyTorch operation:

Problem:
{example['problem_code']}

Requirements:
- Write efficient CUDA kernel code
- Include proper memory management
- Optimize for performance
- Ensure correctness [/INST]

Solution:"""

        target = example['solution_code']

        # Combine prompt and target for causal language modeling
        full_text = prompt + "\n" + target

        # Tokenize
        encoding = self.tokenizer(
            full_text,
            truncation=True,
            max_length=self.max_length,
            padding='max_length',
            return_tensors='pt'
        )

        # For causal LM, input_ids and labels are the same
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': encoding['input_ids'].flatten()
        }


class KernelBenchTrainer:
    """Main trainer class for KernelBench model"""

    def __init__(self, config: ModelTrainingConfig):
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        logger.info(f"Initializing trainer with device: {self.device}")
        logger.info(f"Model: {config.model_name}")

    def load_training_data(self) -> Dict:
        """Load prepared training data"""
        data_file = "training_data/kernelbench_splits.json"

        if not os.path.exists(data_file):
            raise FileNotFoundError(f"Training data not found: {data_file}")

        with open(data_file, 'r') as f:
            data_splits = json.load(f)

        logger.info(f"Loaded training data:")
        logger.info(f"  Train: {len(data_splits['train'])}")
        logger.info(f"  Val: {len(data_splits['val'])}")
        logger.info(f"  Test: {len(data_splits['test'])}")

        return data_splits

    def setup_model_and_tokenizer(self):
        """Set up the model and tokenizer"""
        try:
            from transformers import AutoTokenizer, AutoModelForCausalLM

            logger.info(f"Loading tokenizer and model: {self.config.model_name}")

            # Load tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(self.config.model_name)

            # Add padding token if not exists
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token

            # Load model with 4-bit quantization for memory efficiency
            from transformers import BitsAndBytesConfig

            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True
            )

            self.model = AutoModelForCausalLM.from_pretrained(
                self.config.model_name,
                quantization_config=quantization_config,
                torch_dtype=torch.float16,
                device_map="auto",
                trust_remote_code=True
            )

            logger.info(f"Model loaded successfully")
            logger.info(f"Model parameters: {self.model.num_parameters():,}")

            return True

        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            return False

    def create_dataloaders(self, data_splits: Dict):
        """Create training and validation dataloaders"""
        try:
            from torch.utils.data import DataLoader

            # Create datasets
            train_dataset = KernelBenchDataset(
                data_splits['train'],
                self.tokenizer,
                self.config.max_length
            )

            val_dataset = KernelBenchDataset(
                data_splits['val'],
                self.tokenizer,
                self.config.max_length
            ) if data_splits['val'] else None

            # Create dataloaders
            train_dataloader = DataLoader(
                train_dataset,
                batch_size=self.config.batch_size,
                shuffle=True,
                pin_memory=True
            )

            val_dataloader = DataLoader(
                val_dataset,
                batch_size=self.config.batch_size,
                shuffle=False,
                pin_memory=True
            ) if val_dataset else None

            logger.info(f"Created dataloaders:")
            logger.info(f"  Train batches: {len(train_dataloader)}")
            logger.info(f"  Val batches: {len(val_dataloader) if val_dataloader else 0}")

            return train_dataloader, val_dataloader

        except Exception as e:
            logger.error(f"Failed to create dataloaders: {e}")
            return None, None

    def setup_training_components(self):
        """Set up optimizer, scheduler, and other training components"""
        try:
            from transformers import get_linear_schedule_with_warmup

            # Optimizer
            self.optimizer = torch.optim.AdamW(
                self.model.parameters(),
                lr=self.config.learning_rate,
                weight_decay=0.01
            )

            # Calculate total steps for scheduler
            total_steps = len(self.train_dataloader) * self.config.num_epochs

            # Scheduler
            self.scheduler = get_linear_schedule_with_warmup(
                self.optimizer,
                num_warmup_steps=self.config.warmup_steps,
                num_training_steps=total_steps
            )

            logger.info(f"Training components set up:")
            logger.info(f"  Total steps: {total_steps}")
            logger.info(f"  Warmup steps: {self.config.warmup_steps}")

            return True

        except Exception as e:
            logger.error(f"Failed to setup training components: {e}")
            return False

    def train_epoch(self, epoch: int):
        """Train for one epoch"""
        self.model.train()
        total_loss = 0
        num_batches = len(self.train_dataloader)

        logger.info(f"Starting epoch {epoch + 1}/{self.config.num_epochs}")

        for step, batch in enumerate(self.train_dataloader):
            # Move batch to device
            batch = {k: v.to(self.device) for k, v in batch.items()}

            # Forward pass
            outputs = self.model(**batch)
            loss = outputs.loss

            # Backward pass
            loss.backward()

            # Update weights
            self.optimizer.step()
            self.scheduler.step()
            self.optimizer.zero_grad()

            total_loss += loss.item()

            # Logging
            if step % self.config.logging_steps == 0:
                avg_loss = total_loss / (step + 1)
                logger.info(f"  Step {step}/{num_batches}, Loss: {avg_loss:.4f}")

            # Save checkpoint
            if step % self.config.save_steps == 0 and step > 0:
                self.save_checkpoint(epoch, step)

        avg_epoch_loss = total_loss / num_batches
        logger.info(f"Epoch {epoch + 1} completed. Average loss: {avg_epoch_loss:.4f}")

        return avg_epoch_loss

    def evaluate(self):
        """Evaluate the model on validation set"""
        if not self.val_dataloader:
            logger.info("No validation data available")
            return None

        self.model.eval()
        total_loss = 0
        num_batches = len(self.val_dataloader)

        with torch.no_grad():
            for batch in self.val_dataloader:
                batch = {k: v.to(self.device) for k, v in batch.items()}
                outputs = self.model(**batch)
                total_loss += outputs.loss.item()

        avg_loss = total_loss / num_batches
        logger.info(f"Validation loss: {avg_loss:.4f}")

        return avg_loss

    def save_checkpoint(self, epoch: int, step: int):
        """Save model checkpoint"""
        checkpoint_dir = f"{self.config.output_dir}/checkpoint-epoch-{epoch}-step-{step}"
        os.makedirs(checkpoint_dir, exist_ok=True)

        # Save model and tokenizer
        self.model.save_pretrained(checkpoint_dir)
        self.tokenizer.save_pretrained(checkpoint_dir)

        # Save training state
        training_state = {
            'epoch': epoch,
            'step': step,
            'optimizer_state': self.optimizer.state_dict(),
            'scheduler_state': self.scheduler.state_dict(),
            'config': self.config.__dict__
        }

        torch.save(training_state, f"{checkpoint_dir}/training_state.pt")

        logger.info(f"Checkpoint saved: {checkpoint_dir}")

    def train(self):
        """Main training loop"""
        logger.info("🚀 STARTING KERNELBENCH MODEL TRAINING")
        logger.info("=" * 60)

        # Load data
        data_splits = self.load_training_data()

        # Setup model
        if not self.setup_model_and_tokenizer():
            logger.error("Failed to setup model")
            return False

        # Create dataloaders
        self.train_dataloader, self.val_dataloader = self.create_dataloaders(data_splits)
        if not self.train_dataloader:
            logger.error("Failed to create dataloaders")
            return False

        # Setup training components
        if not self.setup_training_components():
            logger.error("Failed to setup training components")
            return False

        # Training loop
        best_val_loss = float('inf')
        training_history = []

        for epoch in range(self.config.num_epochs):
            # Train epoch
            train_loss = self.train_epoch(epoch)

            # Evaluate
            val_loss = self.evaluate()

            # Record history
            epoch_info = {
                'epoch': epoch + 1,
                'train_loss': train_loss,
                'val_loss': val_loss,
                'timestamp': datetime.now().isoformat()
            }
            training_history.append(epoch_info)

            # Save best model
            if val_loss and val_loss < best_val_loss:
                best_val_loss = val_loss
                self.save_checkpoint(epoch, -1)  # -1 indicates best model
                logger.info(f"New best model saved with val_loss: {val_loss:.4f}")

        # Save final model
        final_dir = f"{self.config.output_dir}/final_model"
        os.makedirs(final_dir, exist_ok=True)
        self.model.save_pretrained(final_dir)
        self.tokenizer.save_pretrained(final_dir)

        # Save training history
        with open(f"{self.config.output_dir}/training_history.json", 'w') as f:
            json.dump(training_history, f, indent=2)

        logger.info("🎉 TRAINING COMPLETED!")
        logger.info(f"Final model saved to: {final_dir}")
        logger.info(f"Best validation loss: {best_val_loss:.4f}")

        return True

    def generate_kernel(self, problem_code: str, max_length: int = 2048) -> str:
        """Generate a kernel solution for a given problem"""
        prompt = f"""<s>[INST] You are an expert CUDA kernel developer. Generate an optimized CUDA kernel solution for this PyTorch operation:

Problem:
{problem_code}

Requirements:
- Write efficient CUDA kernel code
- Include proper memory management
- Optimize for performance
- Ensure correctness [/INST]

Solution:"""

        # Tokenize input
        inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        # Generate
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_length=max_length,
                num_return_sequences=1,
                temperature=0.7,
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id
            )

        # Decode
        generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)

        # Extract solution part
        if "Solution:" in generated_text:
            solution = generated_text.split("Solution:")[-1].strip()
        else:
            solution = generated_text[len(prompt):].strip()

        return solution


def main():
    """Main training execution"""
    logger.info("🔬 KERNELBENCH SPECIALIZED MODEL TRAINING")
    logger.info("=" * 80)

    # Check prerequisites
    try:
        import transformers
        logger.info(f"✅ Transformers version: {transformers.__version__}")
    except ImportError:
        logger.error("❌ Transformers not installed. Run: pip install transformers")
        return

    if not torch.cuda.is_available():
        logger.warning("⚠️ CUDA not available. Training will be slow.")
    else:
        logger.info(f"✅ CUDA available: {torch.cuda.get_device_name()}")

    # Create trainer
    config = ModelTrainingConfig()
    trainer = KernelBenchTrainer(config)

    # Start training
    success = trainer.train()

    if success:
        logger.info("🎯 TRAINING SUMMARY")
        logger.info("=" * 40)
        logger.info(f"✅ Model trained successfully")
        logger.info(f"📁 Model saved to: {config.output_dir}")
        logger.info(f"🔬 Ready for KernelBench evaluation")

        # Test generation
        logger.info("\n🧪 TESTING GENERATION")
        logger.info("-" * 30)

        test_problem = """import torch
import torch.nn as nn

class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()

    def forward(self, x):
        return torch.relu(x)

def get_inputs():
    return [torch.randn(256, 256).cuda()]

def get_init_inputs():
    return []
"""

        logger.info("Test problem:")
        logger.info(test_problem[:100] + "...")

        try:
            solution = trainer.generate_kernel(test_problem)
            logger.info(f"Generated solution ({len(solution)} chars):")
            logger.info(solution[:200] + "..." if len(solution) > 200 else solution)
        except Exception as e:
            logger.error(f"Generation failed: {e}")

    else:
        logger.error("❌ Training failed")


if __name__ == "__main__":
    main()