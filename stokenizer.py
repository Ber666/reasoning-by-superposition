from transformers import PreTrainedTokenizer
from typing import List, Optional, Dict

class STokenizer(PreTrainedTokenizer):
    def __init__(self):
        # Create vocabulary: "0" -> 0, "1" -> 1, ..., "29" -> 29
        self.vocab = {str(i): i for i in range(0, 31)}
        self.vocab['<|start-latent|>'] = 31
        self.vocab['<|end-latent|>'] = 32
        self.vocab['<|latent|>'] = 33
        self.vocab['|'] = 34
        self.vocab['[Q]'] = 35
        self.vocab['[R]'] = 36
        self.vocab['[A]'] = 37
        # Add special tokens
        self.vocab['<eos>'] = 38
        self.vocab['<|no-answer|>'] = 39
        
        # Create inverse vocabulary (id to token mapping)
        self.ids_to_tokens = {v: k for k, v in self.vocab.items()}
        
        # Set special token attributes
        self.pad_token = '<eos>'
        self.eos_token = '<eos>'
        self.bos_token = '<eos>'
        # self.unk_token = '<unk>'
        
        super().__init__()

    def get_vocab(self) -> Dict[str, int]:
        """Returns the vocabulary as a dict"""
        return self.vocab.copy()

    @property
    def vocab_size(self) -> int:
        return len(self.vocab)
        
    def _tokenize(self, text: str) -> List[str]:
        # Split on whitespace and validate each token is a number in range
        tokens = []
        for token in text.replace("\n", " ").strip().split():
            if token in self.vocab:
                tokens.append(token)
            else:
                raise ValueError(f"Token {token} not in vocabulary")
        return tokens
    
    def _convert_token_to_id(self, token: str) -> int:
        # Convert token to id, return unk_token_id if token not in vocab
        return self.vocab[token]
    
    def _convert_id_to_token(self, index: int) -> str:
        # Convert id back to token
        return self.ids_to_tokens[index]
    
    def convert_tokens_to_string(self, tokens: List[str]) -> str:
        # Join tokens with spaces
        return ' '.join(tokens)
    
    def build_inputs_with_special_tokens(self, token_ids_0: List[int], 
                                       token_ids_1: Optional[List[int]] = None) -> List[int]:
        # Add special tokens around sequence(s)
        if token_ids_1 is None:
            return token_ids_0 + [self.vocab['<eos>']]
        return token_ids_0 + [self.vocab['<eos>']] + token_ids_1 + [self.vocab['<eos>']]

    def get_special_tokens_mask(self, token_ids_0: List[int], 
                              token_ids_1: Optional[List[int]] = None,
                              already_has_special_tokens: bool = False) -> List[int]:
        # Return a mask indicating special tokens
        if already_has_special_tokens:
            return [1 if token_id in [self.vocab['<eos>'], self.vocab['<pad>'], self.vocab['<unk>']]
                   else 0 for token_id in token_ids_0]
        if token_ids_1 is None:
            return [0] * len(token_ids_0) + [1]
        return [0] * len(token_ids_0) + [1] + [0] * len(token_ids_1) + [1]
