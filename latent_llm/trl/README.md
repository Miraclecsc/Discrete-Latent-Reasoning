# DLR TRL Fork

This is a minimal local TRL fork used by the DLR training scripts.

DLR-specific files include:

- `trl/models/latent_utils.py`
- `trl/models/local_model_utils.py`
- `trl/trainer/latent_sft_config.py`
- `trl/trainer/latent_sft_trainer.py`
- `trl/trainer/latent_grpo_config.py`
- `trl/trainer/latent_grpo_trainer.py`
- `trl/trainer/latent_trainer_mixin.py`

The package name remains `trl` so the training scripts can import standard TRL components and the latent extensions from the same install.
