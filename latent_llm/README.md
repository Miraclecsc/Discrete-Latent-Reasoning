# Latent LLM Stage

This folder contains the augmented latent language-model training code.

- `scripts/train_latent_pretrain.py`: latent-text alignment / pretraining.
- `scripts/train_latent_sft.py`: teacher-forced latent trajectory SFT.
- `scripts/train_latent_grpo.py`: GRPO training over latent-extended checkpoints.
- `scripts/prepare_*`: dataset conversion utilities.
- `trl/`: local TRL fork with DLR latent-token support.

Install the local TRL fork with:

```bash
pip install -e latent_llm/trl
```
