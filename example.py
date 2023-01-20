from deep_chem_fingerprint import Fingerprinter


fp= Fingerprinter(
    weights_path = "model/encoder-weights.ckpt",
    tokenizer_path = "model/smiles-tokenizer.json",
    n_samples = 32
)

smiles_string = "CCCCCC"
fingerprint = fp(smiles_string)

print(fingerprint.shape)
