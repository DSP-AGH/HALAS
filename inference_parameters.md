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

