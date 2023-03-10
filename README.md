# deep-chem-fingerprint

![image](https://user-images.githubusercontent.com/14167817/214374581-5c5d2c93-e758-40d4-ae2e-334f300cb518.png)

Deep chemical fingerprints (256d embeddings) from a Transformer-based neural network trained to canonicalize 1B Zinc15 SMILES strings.

[[Colab Notebook]](https://colab.research.google.com/drive/15cVZpu7M7-qiH6iSsDcjlCyaB8glvcM3?usp=sharing)

# Usage
```python3
from deep_chem_fingerprint import Fingerprinter


fp = Fingerprinter(
    weights_path = "model/encoder-weights.ckpt",
    tokenizer_path = "model/smiles-tokenizer.json",
    n_samples = 32
)

smiles_string = "CCCCCC"
fingerprint = fp(smiles_string)

print(fingerprint.shape)
```
