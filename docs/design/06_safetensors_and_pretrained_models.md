# Design Memo 06: Loading `.safetensors` & Running a Pretrained Model

> **Status:** Design Memo (not implemented in v0.3.0).

## 1. The Problem
ZEN models currently save/load weights only in a custom JSON format (`save_weights` / `load_weights`), which is verbose, slow on large tensors, and incompatible with open-source pretrained checkpoints distributed in `.safetensors` on Hugging Face.

## 2. Proposed Syntax (10 Lines of Examples)

```zen
import zennum as zn
import zennet as nn
import safetensors as st

// Inspect tensor metadata without loading full weights into RAM
let info = st.inspect("tinyllama_q_proj.safetensors")
say f"tensors={keys(info)}"

// Load weights into ZenNum arrays and bind onto a ZEN model
let weights = st.load("tinyllama_layer.safetensors")
let proj = nn.Linear(64, 64)
proj.weight = nn.tensor(weights["model.layers.0.self_attn.q_proj.weight"].T, false)
```

## 3. Error Messages
- `E070 [line 9, col 15] Invalid safetensors header: header length exceeds file size (corrupt or truncated file).`
- `E071 [line 9, col 15] Unsupported tensor dtype 'BF16' in 'w': add float32 upcast or wait for BF16 conversion.`
- `E072 [line 12, col 1] Shape mismatch loading 'proj.weight': checkpoint has [64, 128], model expects [64, 64].`

## 4. Edge Cases
- **Wire format simplicity:** `.safetensors` is 100% implementable in pure Python stdlib (`struct.unpack("<Q", ...)` for 8-byte header size + `json.loads` header + byte-slice `array('f')` / `struct.iter_unpack`!) — zero external libraries needed!
- **Dtypes (`F16`, `BF16`, `F32`, `F64`, `I64`):** `F32` maps directly to `array('f')`; `F64` to `array('d')`; `BF16` converts in pure Python via bit-shift `(u16 << 16) -> struct.unpack('<f', ...)` and `F16` via `struct.unpack('<e', ...)` upcast to `float32`/`float64`!
- **Security & bounds checks:** Reject negative/overlapping byte `data_offsets`, `header_len > 100 MB`, or offsets past EOF so untrusted files cannot OOM or read arbitrary memory.
- **CPU RAM budget:** Support lazy per-tensor reading (`st.read_tensor(path, name)`) and `F32` mode so low-RAM laptops don't double-buffer huge checkpoints as `float64`.

## 5. Minimal Prototype Plan
1. Add pure-Python `.safetensors` parser/writer in `zen/safetensors_rt.py` (supporting `F32`, `F64`, `F16`, `BF16`, `I64`, `I32`) exposed via `stdlib/safetensors.zen` (`inspect`, `load`, `read_tensor`, `save`).
2. Add `model.load_state_dict(weights, strict = true)` helper on `nn.Module` verifying shapes and copying values cleanly.
3. Build a small end-to-end demo loading a real tiny GPT-2 / SmolLM-style weight slice (exported to `.safetensors`) and running greedy forward inference in ZEN matching reference PyTorch logits within `1e-4`.

## 6. Test Plan
- Write synthetic binary `.safetensors` buffers (`F32`, `F64`, `F16`, `BF16`, `I64`), verify `inspect`, `load`, `save` roundtrip equality.
- Negative tests: truncated header, out-of-bounds `data_offsets`, unknown dtype, and `strict=true` shape mismatch.
- Tiny GPT-2 block forward equivalence test (2-token prompt logits compared against precomputed reference vector stored in test fixture, no `torch` import required).

## 7. Honest Difficulty and Risks
- **Inference speed:** Pure-Python `float64` matmul is too slow for multi-hundred-million-parameter LLMs; running a real pretrained model comfortably (even 10M–135M params) requires C99/BLAS or SIMD `F32` kernels (Memo 01) + tokenizer `vocab.json`/`merges.txt` loader.
- **Weight layout convention:** PyTorch `nn.Linear` stores `[out_features, in_features]` while ZenNet `nn.Linear` stores `[in_features, out_features]`, requiring explicit `.T` on load.

## 8. Effort Estimate
- **6–8 steps:** Pure-Python safetensors reader/writer + bounds checks (2), `F16`/`BF16` unpackers (1), `Module.load_state_dict` (2), `vocab.json` BPE loader + tiny pretrained GPT inference test (3).
