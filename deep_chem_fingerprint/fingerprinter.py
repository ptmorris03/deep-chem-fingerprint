import torch

from pathlib import Path
from typing import Union

from .model import load_pretrained_model
from .tokenizer import SMILESTokenizer


class Fingerprinter:
    def __init__(
        self, 
        weights_path: Path = "model/encoder-weights.ckpt",
        tokenizer_path: Path = "model/smiles-tokenizer.json",
        n_samples: int = 32
        ):

        self.tokenizer = SMILESTokenizer(tokenizer_path)
        self.model = load_pretrained_model(weights_path)
        self.n_samples = n_samples

    def __call__(self, smiles_string: str) -> Union[None, torch.tensor]:
        tokens = self.tokenizer.tokenize(smiles_string, n_samples=self.n_samples)
        
        if tokens is None:
            return None

        with torch.no_grad():
            fingerprints = self.model(tokens)

        fingerprints = fingerprints.mean(axis=0)
        return fingerprints
