from abc import ABC, abstractmethod
from typing import List, Sequence, Dict, Any
import logging

# Set up logging
logger = logging.getLogger('twotower.tokenisers')

class BaseTokeniser(ABC):
    @abstractmethod
    def fit(self, texts: Sequence[str]):
        """Fit the tokeniser on a corpus of texts"""
        pass
    
    @abstractmethod
    def encode(self, text: str) -> List[int]:
        """Convert text to token IDs"""
        pass
    
    @abstractmethod
    def truncate_and_pad(self, sequence: List[int], max_len: int) -> List[int]:
        """Truncate or pad a sequence to a fixed length"""
        pass
    
    @property
    @abstractmethod
    def vocab_size(self) -> int:
        """Return the size of the vocabulary"""
        pass

# Character tokeniser (original implementation)
class CharTokeniser(BaseTokeniser):
    PAD = 0
    
    def __init__(self):
        self.string_to_index = {}
        self.index_to_string = {}
    
    def fit(self, texts: Sequence[str]):
        """Build vocabulary from a corpus of texts"""
        logger.info(f"Building vocabulary from {len(texts)} texts")
        if texts:
            logger.info(f"Sample texts: {texts[:3]}")
        
        # Get all unique characters
        chars = sorted({char for text in texts for char in text})
        logger.info(f"Found {len(chars)} unique characters: {chars[:50]}{'...' if len(chars) > 50 else ''}")
        
        self.string_to_index = {char: idx+1 for idx, char in enumerate(chars)} # 0 = padding
        self.index_to_string = {idx: char for char, idx in self.string_to_index.items()}
        
        logger.info(f"Vocabulary size (including padding): {self.vocab_size}")
        logger.info(f"Sample mappings: {list(self.string_to_index.items())[:5]}")
        return self
    
    def encode(self, text: str) -> List[int]:
        """Convert text to character IDs"""
        encoded = [self.string_to_index.get(char, 0) for char in text]
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(f"Encoded text '{text[:20]}...' to {encoded[:20]}...")
        return encoded
    
    def decode(self, indices: List[int]) -> str:
        """Convert character IDs back to text"""
        decoded = ''.join(self.index_to_string.get(idx, '?') for idx in indices)
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(f"Decoded {indices[:20]}... to '{decoded[:20]}...'")
        return decoded
    
    def truncate_and_pad(self, sequence: List[int], max_len: int) -> List[int]:
        """Truncate or pad a sequence to a fixed length"""
        original_len = len(sequence)
        if len(sequence) < max_len:
            padded = sequence + [self.PAD] * (max_len - len(sequence))
            if logger.isEnabledFor(logging.DEBUG):
                logger.debug(f"Padded sequence from length {original_len} to {max_len}")
            return padded
        else:
            truncated = sequence[:max_len]
            if logger.isEnabledFor(logging.DEBUG):
                logger.debug(f"Truncated sequence from length {original_len} to {max_len}")
            return truncated
    
    @property
    def vocab_size(self) -> int:
        """Return the size of the vocabulary including padding token"""
        return len(self.string_to_index) + 1
    
    def save(self, filepath: str):
        """Save the tokeniser's vocabulary to a file"""
        import pickle
        with open(filepath, 'wb') as f:
            pickle.dump(self.string_to_index, f)
        logger.info(f"Saved tokeniser vocabulary to {filepath}")
    
    @classmethod
    def load(cls, filepath: str):
        """Load a tokeniser's vocabulary from a file"""
        import pickle
        tokeniser = cls()
        with open(filepath, 'rb') as f:
            tokeniser.string_to_index = pickle.load(f)
        tokeniser.index_to_string = {idx: char for char, idx in tokeniser.string_to_index.items()}
        logger.info(f"Loaded tokeniser vocabulary from {filepath} with {tokeniser.vocab_size} tokens")
        return tokeniser

# Registry of available tokenisers
REGISTRY = {
    "char": CharTokeniser,
    # Add more tokenisers here, e.g., WordTokeniser, BPETokeniser, etc.
}

def build(name: str, **kwargs) -> BaseTokeniser:
    """Build a tokeniser by name from the registry"""
    if name not in REGISTRY:
        raise ValueError(f"Unknown tokeniser: {name}. Available options: {list(REGISTRY.keys())}")
    return REGISTRY[name](**kwargs) 