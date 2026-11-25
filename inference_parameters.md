# Inference Parameters and Model Revisions

This document describes the inference parameters used for each model, along with the corresponding model revision hashes.

> **Note:** GitHub hashes refer to repository commit hashes. Hugging Face hashes refer to model revision commits on Hugging Face.

---

# ASR Models

## Whisper (v2, v3, Turbo)


### Parameters

| Parameter                       | Value                               |
| ------------------------------- | ----------------------------------- |
| language                        | `en`                                |
| language detection              | Disabled                            |
| all other parameters            | Default values                      |

### Revision

```text
dd985ac
```

*Source: GitHub commit hash*
https://github.com/openai/whisper
---

## NeMo Canary 1B / Canary 1B Flash

### Parameters

```yaml
beam:
  beam_size: 1
  max_generation_delta: -1
```
All other parameters set to default values.

### Revision

```text
64994b81e0
```

*Source: GitHub commit hash*
https://github.com/NVIDIA-NeMo/NeMo
---

## NeMo Parakeet 2

### Parameters

All parameters set to default values.

### Revision

```text
64994b81e0
```

*Source: GitHub commit hash*
https://github.com/NVIDIA-NeMo/NeMo
---

## CrisperWhisper

### Parameters

All parameters set to default values.

### Revision

```text
ee9dd41
```

*Source: GitHub commit hash*
https://github.com/nyrahealth/CrisperWhisper
---

## Phi-4 Multimodal Instruct

### Inference Configuration

Inference follows the official example provided by Microsoft:

https://huggingface.co/microsoft/Phi-4-multimodal-instruct/blob/main/sample_inference_phi4mm.py

### Revision

```text
33e62acdd07cd7d6635badd529aa0a3467bb9c6a
```

*Source: Hugging Face revision hash*

---

## IBM Granite Speech 3.3 8B

### Inference Configuration

Inference follows the official model documentation:

https://huggingface.co/ibm-granite/granite-speech-3.3-8b

### Revision

```text
3383cfff351908896007f45d0f252ced61c62550
```

*Source: Hugging Face revision hash*

---

## Oruk Fourier-2B (Community Evaluation)

### Parameters

| Parameter | Value |
| --- | --- |
| model | `oruk/oruk-fourier-2b` (CC BY-NC 4.0) |
| precision | `bfloat16` |
| language | `English` (forced) |
| `do_sample` | `False` (published `generation_config.json`) |
| `num_beams` | `3` (published `generation_config.json`) |
| `no_repeat_ngram_size` | `8` (published `generation_config.json`) |
| `max_new_tokens` | `1024` |
| `bad_words_ids` | `32,253` entries (published `generation_config.json`) |

### Revision

```text
8c2a8871f1304aa2f634da4fb95e24438c4945b9
```

*Source: Hugging Face revision hash (`oruk/oruk-fourier-2b`, `model.safetensors` SHA-256 `e64765ffdd90c83704c0951e0d88b7f2ed84499af5c208f7ddf8d922cf08441c`)*

### Paired Base Comparison (`Qwen/Qwen3-ASR-1.7B-hf`)

Evaluated on the same 3,607 HALAS segments (`distil-whisper/earnings22`, `chunked`, `test`) with `bfloat16`, forced `language="English"`, `do_sample=False`, `num_beams=1`, and `max_new_tokens=1024` (see `scripts/eval_oruk_fourier_2b.py` and `results/halas_comparison.csv`).

