#!/usr/bin/env python3
"""Reproducible HALAS evaluation script for oruk/oruk-fourier-2b (and base Qwen3-ASR-1.7B-hf).

Downloads strictly the 7 native BF16 root checkpoint files from `oruk/oruk-fourier-2b`
at pinned commit `8c2a8871f1304aa2f634da4fb95e24438c4945b9`, matches all 3,607
`HALAS_dataset.csv` segments from `distil-whisper/earnings22` (`chunked`, `test` split)
using TorchCodec-free decoding, and computes apples-to-apples corpus WER, n-gram loop
counts, insertion bursts, and `[INAUDIBLE]` non-empty rates alongside all 9 existing
baseline models in `HALAS_dataset.csv`.

Note: Automated metrics (WER, loops, insertion bursts, `[INAUDIBLE]` behavior) are
reported separately from the human span-level `*_label` annotations, which exist only
for the 9 original HALAS paper models.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import subprocess
import time
from pathlib import Path
from typing import Any

import jiwer
import numpy as np
import soundfile as sf
import torch
from datasets import Audio, load_dataset
from huggingface_hub import snapshot_download
from transformers import AutoModelForMultimodalLM, AutoProcessor
from whisper_normalizer.english import EnglishTextNormalizer

MODEL_ID = "oruk/oruk-fourier-2b"
MODEL_REV = "8c2a8871f1304aa2f634da4fb95e24438c4945b9"
BASE_MODEL_ID = "Qwen/Qwen3-ASR-1.7B-hf"
NATIVE_ROOT_FILES = [
    "chat_template.jinja",
    "config.json",
    "generation_config.json",
    "model.safetensors",
    "processor_config.json",
    "tokenizer.json",
    "tokenizer_config.json",
]
EXPECTED_WEIGHT_BYTES = 4_076_193_080
EXPECTED_WEIGHT_SHA256 = "e64765ffdd90c83704c0951e0d88b7f2ed84499af5c208f7ddf8d922cf08441c"


def sha256_file(path: Path) -> tuple[int, str]:
    digest = hashlib.sha256()
    size = 0
    with path.open("rb") as fh:
        while chunk := fh.read(8 * 1024 * 1024):
            size += len(chunk)
            digest.update(chunk)
    return size, digest.hexdigest()


def decode_audio(audio_dict: dict[str, Any]) -> tuple[np.ndarray, float]:
    raw_bytes = audio_dict.get("bytes")
    raw_path = audio_dict.get("path")
    try:
        if raw_bytes:
            wav, sr = sf.read(io.BytesIO(raw_bytes), dtype="float32", always_2d=False)
        else:
            wav, sr = sf.read(str(raw_path), dtype="float32", always_2d=False)
    except Exception:  # noqa: BLE001
        cmd = [
            "ffmpeg", "-hide_banner", "-loglevel", "error",
            "-i", "pipe:0" if raw_bytes else str(raw_path),
            "-f", "f32le", "-ac", "1", "-ar", "16000", "pipe:1",
        ]
        proc = subprocess.run(cmd, input=bytes(raw_bytes) if raw_bytes else None, capture_output=True, check=True)
        wav = np.frombuffer(proc.stdout, dtype=np.float32).copy()
        sr = 16000
    wav = np.asarray(wav, dtype=np.float32)
    if wav.ndim == 2:
        wav = wav.mean(axis=1)
    dur = float(wav.size) / float(sr)
    if sr != 16000:
        import librosa
        wav = librosa.resample(wav, orig_sr=sr, target_sr=16000).astype(np.float32, copy=False)
    return wav, dur


def has_ngram_loop(norm_text: str, min_n: int = 3, min_repeats: int = 3) -> bool:
    toks = norm_text.split()
    if len(toks) < min_n * min_repeats:
        return False
    for n in range(min_n, min(9, len(toks) // min_repeats + 1)):
        for i in range(len(toks) - n * min_repeats + 1):
            pat = toks[i : i + n]
            reps, j = 1, i + n
            while j + n <= len(toks) and toks[j : j + n] == pat:
                reps += 1
                j += n
            if reps >= min_repeats:
                return True
    run = 1
    for i in range(1, len(toks)):
        run = run + 1 if toks[i] == toks[i - 1] else 1
        if run >= 5:
            return True
    return False


def transcribe_segments(
    model_dir: Path,
    items: list[tuple[str, np.ndarray, float]],
    *,
    gen_kwargs: dict[str, Any],
    bs_short: int = 16,
    bs_long: int = 4,
) -> tuple[dict[str, str], dict[str, Any]]:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.bfloat16 if device.type == "cuda" else torch.float32
    processor = AutoProcessor.from_pretrained(str(model_dir), local_files_only=True)
    processor.tokenizer.padding_side = "left"
    model = (
        AutoModelForMultimodalLM.from_pretrained(str(model_dir), local_files_only=True, dtype=dtype)
        .to(device)
        .eval()
    )
    short_idx = sorted([i for i, x in enumerate(items) if x[2] <= 30.0], key=lambda i: items[i][2], reverse=True)
    long_idx = sorted([i for i, x in enumerate(items) if x[2] > 30.0], key=lambda i: items[i][2], reverse=True)
    batches = (
        [long_idx[k : k + bs_long] for k in range(0, len(long_idx), bs_long)]
        + [short_idx[k : k + bs_short] for k in range(0, len(short_idx), bs_short)]
    )
    preds: dict[str, str] = {}
    t0 = time.perf_counter()
    with torch.inference_mode():
        for b in batches:
            inputs = processor.apply_transcription_request(
                audio=[items[i][1] for i in b],
                language=["English"] * len(b),
                return_tensors="pt",
                processor_kwargs={"padding": True},
            ).to(device, model.dtype)
            out_ids = model.generate(**inputs, **gen_kwargs)
            gen_ids = out_ids[:, inputs["input_ids"].shape[1] :]
            texts = processor.decode(gen_ids, return_format="transcription_only")
            for idx, txt in zip(b, texts, strict=True):
                preds[items[idx][0]] = txt.strip()
    elapsed = time.perf_counter() - t0
    total_audio = sum(x[2] for x in items)
    return preds, {
        "segments": len(items),
        "total_audio_seconds": round(total_audio, 3),
        "decode_seconds": round(elapsed, 3),
        "rtfx": round(total_audio / elapsed, 2) if elapsed > 0 else 0.0,
    }


def score_model(rows: list[dict[str, str]], preds: dict[str, str], norm: EnglishTextNormalizer) -> dict[str, Any]:
    def subset(sub: list[dict[str, str]], ref_col: str) -> dict[str, Any]:
        refs_n, hyps_n, loops, bursts = [], [], 0, 0
        for r in sub:
            rn = norm(r[ref_col] or "")
            if not rn:
                continue
            hn = norm(preds.get(r["audio_id"], "") or "")
            refs_n.append(rn)
            hyps_n.append(hn)
            if has_ngram_loop(hn):
                loops += 1
            if len(hn.split()) > max(2 * len(rn.split()) + 10, len(rn.split()) + 20):
                bursts += 1
        w = jiwer.process_words(refs_n, hyps_n)
        ref_w = w.hits + w.substitutions + w.deletions
        return {
            "n_scored": len(refs_n),
            "wer_percent": round(100.0 * (w.substitutions + w.deletions + w.insertions) / ref_w, 2) if ref_w else 0.0,
            "insertion_rate_percent": round(100.0 * w.insertions / ref_w, 2) if ref_w else 0.0,
            "ngram_loop_segments": loops,
            "insertion_burst_segments": bursts,
        }

    test_aud = [r for r in rows if r["split"] == "test" and "[INAUDIBLE]" not in (r["corrected_reference_text"] or "")]
    all_aud = [r for r in rows if "[INAUDIBLE]" not in (r["corrected_reference_text"] or "")]
    inaud = [r for r in rows if "[INAUDIBLE]" in (r["corrected_reference_text"] or "")]
    inaud_nonempty = sum(bool(norm(preds.get(r["audio_id"], "") or "")) for r in inaud)
    return {
        "test_corrected_ref": subset(test_aud, "corrected_reference_text"),
        "test_e22_ref": subset(test_aud, "e22_reference_text"),
        "all_audible_corrected_ref": subset(all_aud, "corrected_reference_text"),
        "inaudible_nonempty_rate_percent": round(100.0 * inaud_nonempty / len(inaud), 2) if inaud else 0.0,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate oruk/oruk-fourier-2b on HALAS.")
    parser.add_argument("--halas-csv", type=Path, default=Path("HALAS_dataset.csv"))
    parser.add_argument("--output-dir", type=Path, default=Path("results"))
    parser.add_argument("--include-base-qwen", action="store_true")
    args = parser.parse_args()

    local_dir = Path(snapshot_download(repo_id=MODEL_ID, revision=MODEL_REV, allow_patterns=NATIVE_ROOT_FILES))
    wb, ws = sha256_file(local_dir / "model.safetensors")
    if wb != EXPECTED_WEIGHT_BYTES or ws != EXPECTED_WEIGHT_SHA256:
        raise RuntimeError(f"Weight integrity mismatch: bytes={wb} sha256={ws}")

    with args.halas_csv.open(encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    target_ids = {r["audio_id"] for r in rows}

    ds = load_dataset("distil-whisper/earnings22", "chunked", split="test").cast_column("audio", Audio(decode=False))
    matched: dict[str, tuple[np.ndarray, float]] = {}
    for row in ds:
        aid = f"{row['segment_id']}_{row['file_id']}.wav"
        if aid in target_ids and aid not in matched:
            matched[aid] = decode_audio(row["audio"])
    if len(matched) != len(target_ids):
        raise RuntimeError(f"Matched {len(matched)}/{len(target_ids)} HALAS segments")

    items = [(r["audio_id"], matched[r["audio_id"]][0], matched[r["audio_id"]][1]) for r in rows]
    f_preds, f_stats = transcribe_segments(local_dir, items, gen_kwargs={"max_new_tokens": 1024})
    norm = EnglishTextNormalizer()
    summary: dict[str, Any] = {
        "model_id": MODEL_ID,
        "revision": MODEL_REV,
        "runtime": {"oruk_fourier_2b": f_stats},
        "models": {"oruk_fourier_2b": score_model(rows, f_preds, norm)},
    }
    if args.include_base_qwen:
        base_dir = Path(snapshot_download(repo_id=BASE_MODEL_ID))
        b_preds, b_stats = transcribe_segments(
            base_dir, items, gen_kwargs={"do_sample": False, "num_beams": 1, "max_new_tokens": 1024}, bs_short=32, bs_long=8
        )
        summary["runtime"]["qwen3_asr_1_7b"] = b_stats
        summary["models"]["qwen3_asr_1_7b"] = score_model(rows, b_preds, norm)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "oruk_fourier_2b_summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
