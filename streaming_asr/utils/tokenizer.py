"""
SentencePiece tokenizer wrapper for decoding RNN-T output tokens.
Loads tokenizer from .nemo archive or standalone .model file.
"""

import os
from typing import List, Optional


class SentencePieceTokenizer:
    """
    Wrapper around SentencePiece for encoding/decoding text.
    Compatible with NeMo's SentencePiece tokenizer format.
    """

    def __init__(self, model_path: str):
        """
        Args:
            model_path: path to .model SentencePiece file
        """
        try:
            import sentencepiece as spm
        except ImportError:
            raise ImportError(
                "sentencepiece is required: pip install sentencepiece"
            )

        if not os.path.isfile(model_path):
            raise FileNotFoundError(f"Tokenizer model not found: {model_path}")

        self.sp = spm.SentencePieceProcessor()
        self.sp.Load(model_path)

        self._vocab_size = self.sp.GetPieceSize()
        self._blank_id = self._vocab_size  # blank is appended after vocab

    @property
    def vocab_size(self) -> int:
        """Vocabulary size excluding blank token."""
        return self._vocab_size

    @property
    def blank_id(self) -> int:
        return self._blank_id

    def encode(self, text: str) -> List[int]:
        """Encode text to token IDs."""
        return self.sp.EncodeAsIds(text)

    def decode(self, token_ids: List[int]) -> str:
        """
        Decode token IDs to text.
        Filters out blank tokens automatically.
        """
        filtered = [t for t in token_ids if t != self._blank_id and t < self._vocab_size]
        if not filtered:
            return ""
        return self.sp.DecodeIds(filtered)

    def decode_tokens(self, token_ids: List[int]) -> List[str]:
        """Decode individual tokens to their string pieces."""
        return [
            self.sp.IdToPiece(t)
            for t in token_ids
            if t != self._blank_id and t < self._vocab_size
        ]

    def id_to_piece(self, token_id: int) -> str:
        if token_id == self._blank_id:
            return "<blank>"
        if token_id >= self._vocab_size:
            return f"<unk:{token_id}>"
        return self.sp.IdToPiece(token_id)
