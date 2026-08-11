---
tags:
- sentence-transformers
- sentence-similarity
- feature-extraction
- generated_from_trainer
- dataset_size:72
- loss:OnlineContrastiveLoss
base_model: sentence-transformers/all-MiniLM-L6-v2
widget:
- source_sentence: what lets someone pay without creating an account
  sentences:
  - 'Codename Clove. Owner: Payments. Feature flag: clove_ga. Rollout: 25%. Tier:
    1. Status: beta.'
  - 'Codename Elm. Owner: Storefront. Feature flag: elm_ga. Rollout: 100%. Tier: 0.
    Status: GA.'
  - 'Codename Aster. Owner: Storefront. Feature flag: aster_ga. Rollout: 100%. Tier:
    0. Status: GA.'
- source_sentence: what holds funds then charges when the order ships
  sentences:
  - 'Codename Basil. Owner: Storefront. Feature flag: basil_ga. Rollout: 50%. Tier:
    1. Status: GA.'
  - 'Codename Elm. Owner: Storefront. Feature flag: elm_ga. Rollout: 100%. Tier: 0.
    Status: GA.'
  - 'Codename Holly. Owner: Payments. Feature flag: holly_ga. Rollout: 100%. Tier:
    0. Status: GA.'
- source_sentence: which project is buy-now-pay-later at checkout
  sentences:
  - 'Codename Fern. Owner: Payments. Feature flag: fern_ga. Rollout: 75%. Tier: 2.
    Status: GA.'
  - 'Codename Fern. Owner: Payments. Feature flag: fern_ga. Rollout: 75%. Tier: 2.
    Status: GA.'
  - 'Codename Iris. Owner: Payments. Feature flag: iris_ga. Rollout: 60%. Tier: 1.
    Status: GA.'
- source_sentence: what sends a basket to another account to settle
  sentences:
  - 'Codename Aster. Owner: Storefront. Feature flag: aster_ga. Rollout: 100%. Tier:
    0. Status: GA.'
  - 'Codename Fern. Owner: Payments. Feature flag: fern_ga. Rollout: 75%. Tier: 2.
    Status: GA.'
  - 'Codename Basil. Owner: Storefront. Feature flag: basil_ga. Rollout: 50%. Tier:
    1. Status: GA.'
- source_sentence: which project is guest checkout with no sign-in
  sentences:
  - 'Codename Aster. Owner: Storefront. Feature flag: aster_ga. Rollout: 100%. Tier:
    0. Status: GA.'
  - 'Codename Elm. Owner: Storefront. Feature flag: elm_ga. Rollout: 100%. Tier: 0.
    Status: GA.'
  - 'Codename Holly. Owner: Payments. Feature flag: holly_ga. Rollout: 100%. Tier:
    0. Status: GA.'
pipeline_tag: sentence-similarity
library_name: sentence-transformers
---

# SentenceTransformer based on sentence-transformers/all-MiniLM-L6-v2

This is a [sentence-transformers](https://www.SBERT.net) model finetuned from [sentence-transformers/all-MiniLM-L6-v2](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2). It maps sentences & paragraphs to a 384-dimensional dense vector space and can be used for retrieval.

## Model Details

### Model Description
- **Model Type:** Sentence Transformer
- **Base model:** [sentence-transformers/all-MiniLM-L6-v2](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2) <!-- at revision 1110a243fdf4706b3f48f1d95db1a4f5529b4d41 -->
- **Maximum Sequence Length:** 256 tokens
- **Output Dimensionality:** 384 dimensions
- **Similarity Function:** Cosine Similarity
- **Supported Modality:** Text
<!-- - **Training Dataset:** Unknown -->
<!-- - **Language:** Unknown -->
<!-- - **License:** Unknown -->

### Model Sources

- **Documentation:** [Sentence Transformers Documentation](https://sbert.net)
- **Repository:** [Sentence Transformers on GitHub](https://github.com/huggingface/sentence-transformers)
- **Hugging Face:** [Sentence Transformers on Hugging Face](https://huggingface.co/models?library=sentence-transformers)

### Full Model Architecture

```
SentenceTransformer(
  (0): Transformer({'transformer_task': 'feature-extraction', 'modality_config': {'text': {'method': 'forward', 'method_output_name': 'last_hidden_state'}}, 'module_output_name': 'token_embeddings', 'architecture': 'BertModel'})
  (1): Pooling({'embedding_dimension': 384, 'pooling_mode': 'mean', 'include_prompt': True})
  (2): Normalize({})
)
```

## Usage

### Direct Usage (Sentence Transformers)

First install the Sentence Transformers library:

```bash
pip install -U sentence-transformers
```
Then you can load this model and run inference.
```python
from sentence_transformers import SentenceTransformer

# Download from the 🤗 Hub
model = SentenceTransformer("sentence_transformers_model_id")
# Run inference
sentences = [
    'which project is guest checkout with no sign-in',
    'Codename Aster. Owner: Storefront. Feature flag: aster_ga. Rollout: 100%. Tier: 0. Status: GA.',
    'Codename Holly. Owner: Payments. Feature flag: holly_ga. Rollout: 100%. Tier: 0. Status: GA.',
]
embeddings = model.encode(sentences)
print(embeddings.shape)
# [3, 384]

# Get the similarity scores for the embeddings
similarities = model.similarity(embeddings, embeddings)
print(similarities)
# tensor([[1.0000, 0.2254, 0.2808],
#         [0.2254, 1.0000, 0.6382],
#         [0.2808, 0.6382, 1.0000]])
```
<!--
### Direct Usage (Transformers)

<details><summary>Click to see the direct usage in Transformers</summary>

</details>
-->

<!--
### Downstream Usage (Sentence Transformers)

You can finetune this model on your own dataset.

<details><summary>Click to expand</summary>

</details>
-->

<!--
### Out-of-Scope Use

*List how the model may foreseeably be misused and address what users ought not to do with the model.*
-->

<!--
## Bias, Risks and Limitations

*What are the known or foreseeable issues stemming from this model? You could also flag here known failure cases or weaknesses of the model.*
-->

<!--
### Recommendations

*What are recommendations with respect to the foreseeable issues? For example, filtering explicit content.*
-->

## Training Details

### Training Dataset

#### Unnamed Dataset

* Size: 72 training samples
* Columns: <code>sentence1</code>, <code>sentence2</code>, and <code>label</code>
* Approximate statistics based on the first 72 samples:
  |          | sentence1                                                                         | sentence2                                                                          | label                                           |
  |:---------|:----------------------------------------------------------------------------------|:-----------------------------------------------------------------------------------|:------------------------------------------------|
  | type     | string                                                                            | string                                                                             | int                                             |
  | modality | text                                                                              | text                                                                               |                                                 |
  | details  | <ul><li>min: 9 tokens</li><li>mean: 12.33 tokens</li><li>max: 16 tokens</li></ul> | <ul><li>min: 31 tokens</li><li>mean: 32.33 tokens</li><li>max: 34 tokens</li></ul> | <ul><li>0: ~75.00%</li><li>1: ~25.00%</li></ul> |
* Samples:
  | sentence1                                                                   | sentence2                                                                                                   | label          |
  |:----------------------------------------------------------------------------|:------------------------------------------------------------------------------------------------------------|:---------------|
  | <code>which project is the one-tap reorder flow for returning buyers</code> | <code>Codename Aster. Owner: Storefront. Feature flag: aster_ga. Rollout: 100%. Tier: 0. Status: GA.</code> | <code>1</code> |
  | <code>which project is the one-tap reorder flow for returning buyers</code> | <code>Codename Clove. Owner: Payments. Feature flag: clove_ga. Rollout: 25%. Tier: 1. Status: beta.</code>  | <code>0</code> |
  | <code>which project is the one-tap reorder flow for returning buyers</code> | <code>Codename Basil. Owner: Storefront. Feature flag: basil_ga. Rollout: 50%. Tier: 1. Status: GA.</code>  | <code>0</code> |
* Loss: [<code>OnlineContrastiveLoss</code>](https://sbert.net/docs/package_reference/sentence_transformer/losses.html#onlinecontrastiveloss) with these parameters:
  ```json
  {
      "distance_metric": "SiameseDistanceMetric.COSINE_DISTANCE",
      "margin": 0.5
  }
  ```

### Training Hyperparameters
#### Non-Default Hyperparameters

- `per_device_train_batch_size`: 16
- `num_train_epochs`: 1
- `learning_rate`: 2e-05
- `warmup_steps`: 10

#### All Hyperparameters
<details><summary>Click to expand</summary>

- `per_device_train_batch_size`: 16
- `num_train_epochs`: 1
- `max_steps`: -1
- `learning_rate`: 2e-05
- `lr_scheduler_type`: linear
- `lr_scheduler_kwargs`: None
- `warmup_steps`: 10
- `optim`: adamw_torch_fused
- `optim_args`: None
- `weight_decay`: 0.0
- `adam_beta1`: 0.9
- `adam_beta2`: 0.999
- `adam_epsilon`: 1e-08
- `optim_target_modules`: None
- `gradient_accumulation_steps`: 1
- `average_tokens_across_devices`: True
- `max_grad_norm`: 1.0
- `label_smoothing_factor`: 0.0
- `bf16`: False
- `fp16`: False
- `bf16_full_eval`: False
- `fp16_full_eval`: False
- `tf32`: None
- `gradient_checkpointing`: False
- `gradient_checkpointing_kwargs`: None
- `torch_compile`: False
- `torch_compile_backend`: None
- `torch_compile_mode`: None
- `use_liger_kernel`: False
- `liger_kernel_config`: None
- `use_cache`: False
- `neftune_noise_alpha`: None
- `torch_empty_cache_steps`: None
- `auto_find_batch_size`: False
- `log_on_each_node`: True
- `logging_nan_inf_filter`: True
- `include_num_input_tokens_seen`: no
- `log_level`: passive
- `log_level_replica`: warning
- `disable_tqdm`: False
- `project`: huggingface
- `trackio_space_id`: None
- `trackio_bucket_id`: None
- `trackio_static_space_id`: None
- `per_device_eval_batch_size`: 8
- `prediction_loss_only`: True
- `eval_on_start`: False
- `eval_do_concat_batches`: True
- `eval_use_gather_object`: False
- `eval_accumulation_steps`: None
- `include_for_metrics`: []
- `batch_eval_metrics`: False
- `save_only_model`: False
- `save_on_each_node`: False
- `enable_jit_checkpoint`: False
- `push_to_hub`: False
- `hub_private_repo`: None
- `hub_model_id`: None
- `hub_strategy`: every_save
- `hub_always_push`: False
- `hub_revision`: None
- `load_best_model_at_end`: False
- `ignore_data_skip`: False
- `restore_callback_states_from_checkpoint`: False
- `full_determinism`: False
- `seed`: 42
- `data_seed`: None
- `use_cpu`: False
- `accelerator_config`: {'split_batches': False, 'dispatch_batches': None, 'even_batches': True, 'use_seedable_sampler': True, 'non_blocking': False, 'gradient_accumulation_kwargs': None}
- `parallelism_config`: None
- `dataloader_drop_last`: False
- `dataloader_num_workers`: 0
- `dataloader_pin_memory`: True
- `dataloader_persistent_workers`: False
- `dataloader_prefetch_factor`: None
- `remove_unused_columns`: True
- `label_names`: None
- `train_sampling_strategy`: random
- `length_column_name`: length
- `ddp_find_unused_parameters`: None
- `ddp_bucket_cap_mb`: None
- `ddp_broadcast_buffers`: False
- `ddp_static_graph`: None
- `ddp_backend`: None
- `ddp_timeout`: 1800
- `fsdp`: None
- `fsdp_config`: None
- `deepspeed`: None
- `debug`: []
- `skip_memory_metrics`: True
- `do_predict`: False
- `resume_from_checkpoint`: None
- `warmup_ratio`: None
- `local_rank`: -1
- `prompts`: None
- `batch_sampler`: batch_sampler
- `multi_dataset_batch_sampler`: proportional
- `router_mapping`: {}
- `learning_rate_mapping`: {}

</details>

### Training Time
- **Training**: 7.7 seconds

### Framework Versions
- Python: 3.12.9
- Sentence Transformers: 5.6.0
- Transformers: 5.13.0
- PyTorch: 2.12.1+cpu
- Accelerate: 1.14.0
- Datasets: 5.0.0
- Tokenizers: 0.22.2

## Citation

### BibTeX

#### Sentence Transformers
```bibtex
@inproceedings{reimers-2019-sentence-bert,
    title = "Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks",
    author = "Reimers, Nils and Gurevych, Iryna",
    booktitle = "Proceedings of the 2019 Conference on Empirical Methods in Natural Language Processing",
    month = "11",
    year = "2019",
    publisher = "Association for Computational Linguistics",
    url = "https://arxiv.org/abs/1908.10084",
}
```

<!--
## Glossary

*Clearly define terms in order to be accessible across audiences.*
-->

<!--
## Model Card Authors

*Lists the people who create the model card, providing recognition and accountability for the detailed work that goes into its construction.*
-->

<!--
## Model Card Contact

*Provides a way for people who have updates to the Model Card, suggestions, or questions, to contact the Model Card authors.*
-->