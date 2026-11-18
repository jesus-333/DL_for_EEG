# `config/model/hvEEGNet_cfg.py` — `hvEEGNet_config`

Source: `src/jade/config/model/hvEEGNet_cfg.py`

Config dataclass of [`hvEEGNet`](../../model/hvEEGNet.md), a `model_config` subclass with `task = 'generative'` (ClassVar), registered under `hveegnet` and `hierarchical_veegnet` in `MODEL_CONFIG_REGISTRY` (the name is lower-cased, so `"hvEEGNet"` works too). Skeleton TOML: `config/default_config/model/hvEEGNet.toml` (its values equal `hvEEGNet_config()`).

Structure mirrors [`vEEGNet_config`](vEEGNet_cfg.md): flat encoder fields (same names as `EEGNet_config`), `upsample_mode`, `input_size`, `print_var`, and the shared `[pooling]` section (`eegnet_pooling_config`), plus `to_encoder_config()` (forces `name = 'eegnet'`, `flatten_output = False`).

**Differences from `vEEGNet_config`:**
- `parameters_map_type` is `"convolutional"` (default, the paper — each latent space has the shape of its encoder block), `"linear"` or `"lora"` (vector latent spaces of `hidden_space_dimension_list = (32, 128, 512)`, deepest first; `lora_rank = 16`; `use_spectral_norm = False`), mirroring the original `parameters_map_type = 0 / 1 / 2`;
- `use_activation_in_sampling = True` — activation before every latent parameter map (and after the vector projections), as in the original code;
- `use_h_in_decoder = False` — if True, a trainable `h` (NVAE Fig. 2) is combined with the deepest sample; otherwise zeros (the trained model of the paper);
- **defaults are the paper's** (BCI IV-2a, 250 Hz): `c_kernel_1 = (1, 128)`, `c_kernel_3 = (1, 32)`, `use_dropout_2d = True`, pooling stage 1 = `identity` (factor 1), stage 2 = `default` avg pool with factor **10**. Stage 1 is disabled through the `identity` type rather than a negative factor so building the defaults emits no warning.
- the "T too short for the pooling" check skips `identity`/`none` stages as well as negative factors.

Validation is otherwise the same as vEEGNet (kernels coerced to tuples, `c_kernel_2[0] <= C`, `input_size[1:] == (depth_first_layer, C, T)`, etc.).
