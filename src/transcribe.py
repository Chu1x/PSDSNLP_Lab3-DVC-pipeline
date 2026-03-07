import os
import sys
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
if sys.platform == "darwin":
    os.environ["PHONEMIZER_ESPEAK_LIBRARY"] = "/opt/homebrew/lib/libespeak-ng.dylib"

import yaml
import json
import torch
import librosa
import numpy as np
from pathlib import Path
from transformers import Wav2Vec2Processor, Wav2Vec2ForCTC
from utils import write_manifest

def load_model(model_name):
    print(f"Loading model: {model_name}")
    processor = Wav2Vec2Processor.from_pretrained(model_name)
    model = Wav2Vec2ForCTC.from_pretrained(model_name)
    
    # Use GPU if available (CUDA for Linux/Win, MPS for Mac), else CPU
    device = "cpu"
    if torch.cuda.is_available():
        device = "cuda"
    elif torch.backends.mps.is_available():
        device = "mps"
        
    print(f"Using device: {device}")
    model.to(device)
    return processor, model, device

def transcribe_file(wav_path, processor, model, device):
    # Load audio
    # Ensure 16kHz as required by Wav2Vec2
    speech, _ = librosa.load(wav_path, sr=16000)
    
    # Process inputs
    input_values = processor(speech, return_tensors="pt", sampling_rate=16000).input_values
    input_values = input_values.to(device)

    # Inference
    with torch.no_grad():
        logits = model(input_values).logits

    # Decode
    predicted_ids = torch.argmax(logits, dim=-1)
    transcription = processor.batch_decode(predicted_ids)
    
    return transcription[0]

def process_transcription(lang, params):
    model_name = params['transcribe']['model_name']
    input_dirs = params['transcribe']['input_dirs']
    output_dir = Path(params['transcribe']['output_dir'])

    processor, model, device = load_model(model_name)

    # Collect all manifest files from input directories
    manifest_files = []
    for d in input_dirs:
        path = Path(d) / lang
        if path.exists():
            manifest_files.extend(list(path.glob("*.jsonl")))

    if not manifest_files:
        print(f"No manifests found for language {lang}")
        return

    for manifest_path in manifest_files:
        print(f"Processing {manifest_path.name}...")
        
        # Prepare output path: data/predictions/en/clean.jsonl or snr20.jsonl
        out_subdir = output_dir / lang
        out_subdir.mkdir(parents=True, exist_ok=True)
        out_file = out_subdir / manifest_path.name

        # Skip if already exists? (Optional, but DVC handles this usually)
        # For now, we overwrite to ensure freshness
        
        records = []
        with open(manifest_path, 'r', encoding='utf-8') as f:
            for line in f:
                record = json.loads(line)
                wav_path = record['wav_path']
                
                # Transcribe
                try:
                    pred_phon = transcribe_file(wav_path, processor, model, device)
                    
                    # Add prediction to record
                    record['pred_phon'] = pred_phon
                    records.append(record)
                except Exception as e:
                    print(f"Error processing {wav_path}: {e}")

        # Write result
        write_manifest(records, out_file)

def main():
    with open("params.yaml", 'r') as f:
        params = yaml.safe_load(f)
        
    languages = params['base']['languages']
    
    for lang in languages:
        process_transcription(lang, params)

if __name__ == "__main__":
    main()