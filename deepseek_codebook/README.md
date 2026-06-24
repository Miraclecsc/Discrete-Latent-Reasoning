# DeepSeek Codebook Stage

This folder contains the rendered-CoT compression stage used by DLR.

- `train_codebook.py`: trains the codebook-augmented DeepSeek-OCR2 model.
- `export_codebook_ids.py`: exports nearest codebook assignments for rendered examples.
- `visualize_latent_ids.py`: decodes latent ids with the trained OCR decoder for inspection.
- `model_source_10k/`: DeepSeek-OCR2 model source with the stochastic codebook logic.
- `hf_code/`: compatibility source files for local DeepSeek checkpoints.

No checkpoints or rendered datasets are included.
