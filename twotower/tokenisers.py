from abc import ABC, abstractmethod
from typing import List, Sequence, Dict, Any
import logging
import re
import torch

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


class WordTokeniser(BaseTokeniser):
    """Word-level tokeniser with support for lowercase and punctuation removal."""
    PAD = 0
    UNK = 1  # Unknown word token
    
    def __init__(self, lowercase: bool = True, strip_punctuation: bool = True, max_len: int = 32):
        self.word_to_index = {}
        self.index_to_word = {}
        self.lowercase = lowercase
        self.strip_punctuation = strip_punctuation
        self.max_len = max_len
        self._vocab_size = 2  # PAD and UNK tokens
        
        # For compatibility with CharTokeniser
        self.string_to_index = {}
        self.index_to_string = {}
        
        # Regex for tokenization/punctuation removal
        self.word_pattern = re.compile(r'\b\w+\b')
    
    def _preprocess_text(self, text: str) -> str:
        """Apply preprocessing steps to text."""
        if self.lowercase:
            text = text.lower()
        return text
    
    def _tokenize(self, text: str) -> List[str]:
        """Split text into words, optionally removing punctuation."""
        text = self._preprocess_text(text)
        if self.strip_punctuation:
            # Extract only words, removing punctuation
            words = self.word_pattern.findall(text)
        else:
            # Simple whitespace tokenization
            words = text.split()
        return words
    
    def fit(self, texts: Sequence[str]):
        """Build vocabulary from a corpus of texts."""
        logger.info(f"Building word vocabulary from {len(texts)} texts")
        if texts:
            logger.info(f"Sample texts: {texts[:3]}")
        
        # Initialize with special tokens
        self.word_to_index = {'<PAD>': self.PAD, '<UNK>': self.UNK}
        index = len(self.word_to_index)
        
        # Process all texts
        word_counts = {}
        total_words = 0
        
        for text in texts:
            words = self._tokenize(text)
            total_words += len(words)
            for word in words:
                word_counts[word] = word_counts.get(word, 0) + 1
        
        # Sort by frequency
        sorted_words = sorted(word_counts.items(), key=lambda x: x[1], reverse=True)
        logger.info(f"Found {len(sorted_words)} unique words in corpus with {total_words} total words")
        
        # Add to vocabulary
        for word, count in sorted_words:
            self.word_to_index[word] = index
            index += 1
        
        # Create reverse mapping
        self.index_to_word = {idx: word for word, idx in self.word_to_index.items()}
        self._vocab_size = len(self.word_to_index)
        
        # For compatibility with CharTokeniser
        self.string_to_index = self.word_to_index
        self.index_to_string = self.index_to_word
        
        logger.info(f"Final vocabulary size (including special tokens): {self.vocab_size}")
        logger.info(f"Sample mappings: {list(self.word_to_index.items())[:10]}")
        return self
    
    def encode(self, text: str) -> List[int]:
        """Convert text to word IDs."""
        words = self._tokenize(text)
        encoded = [self.word_to_index.get(word, self.UNK) for word in words]
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(f"Encoded text '{text[:30]}...' to {encoded[:10]}...")
        return encoded
    
    def decode(self, indices: List[int]) -> str:
        """Convert word IDs back to text."""
        words = [self.index_to_word.get(idx, '<UNK>') for idx in indices if idx != self.PAD]
        decoded = ' '.join(words)
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(f"Decoded {indices[:10]}... to '{decoded[:30]}...'")
        return decoded
    
    def truncate_and_pad(self, sequence: List[int], max_len: int = None) -> List[int]:
        """Truncate or pad a sequence to a fixed length."""
        if max_len is None:
            max_len = self.max_len
            
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
        """Return the size of the vocabulary including special tokens."""
        return self._vocab_size
    
    def save(self, filepath: str):
        """Save the tokeniser's vocabulary to a file."""
        import pickle
        with open(filepath, 'wb') as f:
            data = {
                'word_to_index': self.word_to_index,
                'lowercase': self.lowercase,
                'strip_punctuation': self.strip_punctuation,
                'max_len': self.max_len
            }
            pickle.dump(data, f)
        logger.info(f"Saved word tokeniser to {filepath}")
    
    @classmethod
    def load(cls, filepath: str):
        """Load a tokeniser's vocabulary from a file."""
        import pickle
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
        
        tokeniser = cls(
            lowercase=data.get('lowercase', True),
            strip_punctuation=data.get('strip_punctuation', True),
            max_len=data.get('max_len', 32)
        )
        tokeniser.word_to_index = data['word_to_index']
        tokeniser.index_to_word = {idx: word for word, idx in tokeniser.word_to_index.items()}
        tokeniser._vocab_size = len(tokeniser.word_to_index)
        
        # For compatibility with CharTokeniser
        tokeniser.string_to_index = tokeniser.word_to_index
        tokeniser.index_to_string = tokeniser.index_to_word
        
        logger.info(f"Loaded word tokeniser from {filepath} with {tokeniser.vocab_size} tokens")
        return tokeniser
    
    def __call__(self, texts):
        """Process a batch of texts."""
        if isinstance(texts, str):
            texts = [texts]
        
        # Encode each text
        encoded = [self.encode(text) for text in texts]
        
        # Pad/truncate to the same length
        padded = [self.truncate_and_pad(e) for e in encoded]
        
        return torch.tensor(padded)


# Registry of available tokenisers
REGISTRY = {
    "char": CharTokeniser,
    "word": WordTokeniser,
    # Add more tokenisers here, e.g., BPETokeniser, etc.
}

def build(name: str, **kwargs) -> BaseTokeniser:
    """Build a tokeniser by name from the registry"""
    if name not in REGISTRY:
        raise ValueError(f"Unknown tokeniser: {name}. Available options: {list(REGISTRY.keys())}")
    return REGISTRY[name](**kwargs) 