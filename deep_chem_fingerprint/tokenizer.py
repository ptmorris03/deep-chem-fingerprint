from rdkit import Chem
from transformers import PreTrainedTokenizerFast
import numpy as np

from pathlib import Path
from typing import Union, List


class SMILESTokenizer:
    def __init__(self, path: Path, max_length: int = 64):
        self.tokenizer = self.load_tokenizer(path)
        self.max_length = max_length

    def load_tokenizer(self, path: Path) -> PreTrainedTokenizerFast:
        tokenizer = PreTrainedTokenizerFast(tokenizer_file=path)
        tokenizer.add_special_tokens({'pad_token': '[PAD]'})
        return tokenizer

    def randomize(self, smiles_string: str, n_samples: int = 32) -> List[str]:
        smiles_strings = []
        try:
            mol = Chem.MolFromSmiles(smiles_string)
            for i in range(n_samples):
                random_string = self.randomize_mol(mol)
                smiles_strings.append(random_string)
            assert len(smiles_strings) > 0
        except AssertionError as e:
            pass
        return smiles_strings

    def randomize_mol(self, mol: Chem.rdchem.Mol) -> str:
        random_string = Chem.MolToSmiles(mol, doRandom=True, canonical=False)
        assert type(random_string) == str
        return random_string

    def tokenize(self, smiles_string: str, n_samples: int = 32) -> np.ndarray:
        random_strings = self.randomize(smiles_string, n_samples)
        return self.tokenizer(
            random_strings,
            return_tensors="pt",
            padding="max_length",
            max_length=self.max_length,
            truncation=True,
            return_token_type_ids=False,
            add_special_tokens=True,
            return_attention_mask=False
        )['input_ids']

    

