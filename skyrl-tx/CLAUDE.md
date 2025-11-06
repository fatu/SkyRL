# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

SkyRL tx is a JAX-based library providing a unified REST API for training and inference of transformer models with multi-adapter LoRA support. It implements a Tinker-compatible API and uses FastAPI with a background processing engine for efficient multi-tenant RL training.

**Key insight**: This project uses a shared base model with multiple concurrent LoRA adapters, enabling many policies to train simultaneously with minimal overhead.

## Development Commands

### Setup
```bash
# Install uv package manager
curl -LsSf https://astral.sh/uv/install.sh | sh

# Install dependencies
uv sync --extra dev --extra tinker

# For GPU support
uv sync --extra gpu

# For TPU support
uv sync --extra tpu
```

### Testing
```bash
# Run all tests
uv run --extra dev --extra tinker pytest -v

# Run specific test
uv run --extra dev --extra tinker pytest -v -s tests/models/test_qwen3_generate.py::test_qwen3_generate_speed

# Run specific test file
uv run --extra dev --extra tinker pytest -v tests/tinker/
```

### Training
```bash
# Basic training command (GPU example with 8 GPUs)
uv run --extra gpu --with jinja2 tx train \
  --model Qwen/Qwen3-4B \
  --dataset HuggingFaceH4/ultrachat_200k \
  --loader tx.loaders.chat \
  --split train_sft \
  --output-dir /tmp/ultrachat \
  --batch-size 8 \
  --load-checkpoint-path /tmp/qwen3 \
  --tp-size 8

# TPU training (same command, just use --extra tpu instead)
uv run --extra tpu --with jinja2 tx train [same args]
```

### Other Commands
```bash
# Show version
tx version

# Build documentation
mkdocs serve
```

## Architecture Overview

### Core Components

**Entry Point**: `tx/run/main.py` - Simple Typer CLI with `train` and `version` commands

**Training Flow** (`tx/run/train.py`):
- Loads HuggingFace models, creates JAX mesh for tensor/data parallelism
- Uses Flax NNX for model definition
- JIT-compiled train_step with value_and_grad
- Supports pluggable data loaders (`tx.loaders`)
- Checkpoint saving with safetensors

**Tinker REST API** (`tx/tinker/api.py`):
- FastAPI server with dual-process architecture:
  - API Process: Handles HTTP, database writes
  - Background Engine: Processes requests asynchronously
- Database-backed request queue (SQLModel/SQLite)
- Key endpoints: `create_model`, `forward_backward`, `optim_step`, `save_weights`, `asample`

**Background Engine** (`tx/tinker/engine.py`):
- Single shared base model with multiple LoRA adapters (up to `max_lora_adapters`)
- Look-ahead scheduling: batches compatible forward_backward operations
- Gradient accumulation per-adapter across batches
- Dynamic batching by model/checkpoint compatibility
- JIT compilation caching with sequence length bucketing
- Polls database every 100ms for pending requests

**Models** (`tx/models/`):
- Currently supports: Qwen3ForCausalLM (including Qwen3Moe variant), MNIST
- Transformer with RoPE, Grouped Query Attention, RMSNorm
- All linear layers support LoRA adapters

**LoRA Implementation** (`tx/layers/lora.py`):
- Multi-adapter design: `lora_A[adapter_idx, in_dim, rank]`, `lora_B[adapter_idx, rank, out_dim]`
- Uses JAX's `ragged_dot` for efficient batched computation
- Per-adapter rank and scaling configuration
- Token-level adapter routing with sorting/unsorting
- Formula: `output = base_output + (x @ lora_A @ lora_B) * scaling`

**Data Loaders** (`tx/loaders/`):
- `text`: Basic text completion (shift-by-1 training)
- `chat`: Chat format with template support
- Pluggable via `--loader` CLI argument

**Utilities** (`tx/utils/`):
- `models.py`: Checkpoint loading/saving (safetensors, PEFT format), adapter extraction/insertion
- `generator.py`: Autoregressive generation with KV caching, temperature sampling, stop tokens
- `storage.py`: Cloud storage support, checkpoint compression (tar.gz)

### Key Design Patterns

**Multi-Adapter LoRA Pattern**:
- Single base model serves multiple training runs
- Adapters indexed 0 to `max_lora_adapters-1` (index 0 reserved for base model inference)
- Batched computation with ragged operations

**Future-Based Async Pattern**:
- Requests create database futures
- Engine polls and processes asynchronously
- Clients poll `/retrieve_future` for results
- 5-minute timeout with 100ms polling

**Checkpoint Dual-Format**:
- Training checkpoints: LoRA weights + optimizer state (msgpack) at `tinker://model_id/weights/checkpoint_id`
- Sampler checkpoints: LoRA weights only (safetensors + PEFT config) at `tinker://model_id/checkpoint_id`

**Gradient Accumulation**:
- `forward_backward` accumulates per-adapter gradients
- `optim_step` applies mean gradient and resets accumulator
- Supports different batch sizes per adapter in same batch

**JAX Sharding**:
- Tensor parallelism across "tp" dimension
- Data parallelism across "dp" dimension
- LoRA params: adapters unsharded, other dims follow base layer

### Request Processing Flow

**Training**:
1. Client calls `/create_model` (assigns adapter index)
2. Client sends batches via `/forward_backward` (accumulates gradients)
3. Engine batches compatible requests together
4. Gradients accumulated per-adapter using adapter_index routing
5. Client calls `/optim_step` (applies Adam update to mean gradient)
6. **Important**: `optim_step` acts as barrier - no `forward_backward` can pass it in queue
7. Client saves via `/save_weights` (training) or `/save_weights_for_sampler` (inference)

**Inference**:
- Base model: `/asample` with `base_model` specified (uses adapter_index=0)
- LoRA adapter: `/asample` with `model_path=tinker://model_id/checkpoint_id`
- Generation: Prefill (process prompt, build KV cache) → Decode (autoregressive)

### Loss Functions

Three types supported (`tx/tinker/loss_fns.py`):
- `cross_entropy`: Standard NLL loss
- `importance_sampling`: Ratio-weighted loss for off-policy RL
- `ppo`: Clipped importance sampling (PPO-style)

Uses `jax.lax.switch` for dynamic per-example loss selection.

## Important Development Considerations

1. **Shared Base Model**: Never create multiple base models for different training runs. Use the multi-adapter LoRA system.

2. **Adapter Indices are Limited**: Only `max_lora_adapters` available (default 32), and index 0 is reserved for base model inference.

3. **Request Ordering Matters**: `optim_step` and `load_weights` act as barriers in the queue for their `model_id`.

4. **Two Checkpoint Formats**: Training checkpoints (with optimizer state) vs sampler checkpoints (inference-ready). Different paths and formats.

5. **JAX JIT Caching**: Sequence lengths are bucketed to ~2 significant bits to reduce compilation overhead. See `round_up_seq_len` in `tx/utils/models.py`.

6. **Dynamic Batching**: The engine intelligently batches compatible requests, even across different `model_id`s.

7. **Gradient Accumulation**: Multiple `forward_backward` calls accumulate, `optim_step` takes the mean. This is per-adapter.

8. **Database-Driven Architecture**: API writes to DB, engine polls DB. No direct inter-process communication.

9. **LoRA Uses ragged_dot**: All LoRA layers use `ragged_dot` for efficient batched computation with variable adapter indices.

10. **KV Cache Pre-allocation**: Padded to `max_length` for fixed memory usage during generation.

## File Structure

```
tx/
├── run/           # CLI entry point and training loop
├── tinker/        # REST API server and background engine
├── models/        # Model architectures (Qwen3, etc.)
├── layers/        # LoRA implementation
├── loaders/       # Data loaders (text, chat)
└── utils/         # Utilities (checkpoints, generation, storage)

tests/
├── models/        # Model tests
├── tinker/        # API tests
└── utils/         # Utility tests
```

## Configuration

Key `EngineConfig` fields:
- `base_model`: HuggingFace model ID
- `max_lora_adapters`: Maximum concurrent adapters (default: 32)
- `max_lora_rank`: Maximum rank per adapter (default: 32)
- `tensor_parallel_size`: TP degree
- `train_micro_batch_size`: Gradient accumulation micro-batch size
- `sample_max_num_sequences`: Inference batch size limit
- `gradient_checkpointing`: Enable activation recomputation
- `shard_attention_heads`: Whether to shard attention across TP

Supports environment variables (e.g., `TX_DATABASE_URL`).

## Related Resources

- Initial blog post: https://novasky-ai.notion.site/skyrl-tx
- v0.1.0 release: https://novasky-ai.notion.site/skyrl-tx-v010
- Ray Summit talk: https://docs.google.com/presentation/d/1g-u8zxz7FsnlQXXShBVoqjUJhS48c6rxkJJJn0sj78A/view
