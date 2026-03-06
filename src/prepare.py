import os
import sys
import yaml
import json
import hashlib
import soundfile as sf
from pathlib import Path
from phonemizer import phonemize
from phonemizer.backend import EspeakBackend

# ==========================================
# macOS Compatibility Fix
# ==========================================
# Explicitly set the path to the espeak-ng library for Homebrew on Apple Silicon.
# If running on Linux/Windows, this might not be needed or paths might differ.
if sys.platform == "darwin":
    os.environ["PHONEMIZER_ESPEAK_LIBRARY"] = "/opt/homebrew/lib/libespeak-ng.dylib"
# ==========================================


def get_audio_md5(file_path):
    """
    Calculates the MD5 checksum of a file.
    Args:
        file_path (Path): Path to the file.
    Returns:
        str: Hex digest of the MD5 hash.
    """
    hash_md5 = hashlib.md5()
    with open(file_path, "rb") as f:
        # Read in chunks to handle large files efficiently
        for chunk in iter(lambda: f.read(4096), b""):
            hash_md5.update(chunk)
    return hash_md5.hexdigest()


def write_manifest(data, output_path):
    """
    Writes a list of dictionaries to a JSONL file atomically.
    Args:
        data (list): List of dictionaries (records).
        output_path (Path): Target file path.
    """
    # Ensure parent directory exists
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Write to a temporary file first
    temp_path = output_path.with_suffix('.tmp')
    with open(temp_path, 'w', encoding='utf-8') as f:
        for record in data:
            # json.dump writes a JSON object; adding \n makes it JSONL
            f.write(json.dumps(record, ensure_ascii=False) + '\n')
    
    # Atomic move: rename temp file to final file
    # This prevents partial writes from corrupting the pipeline state
    if temp_path.exists():
        os.replace(temp_path, output_path)
        print(f"Created manifest: {output_path}")


def process_language(lang, raw_dir, out_dir):
    """
    Process raw data for a specific language and generate a clean manifest.
    """
    lang_dir = raw_dir / lang
    wav_dir = lang_dir / "wav"
    txt_dir = lang_dir / "txt"

    if not wav_dir.exists() or not txt_dir.exists():
        print(f"Warning: Data directories not found for language '{lang}'. Skipping.")
        return

    records = []
    # Glob all .wav files
    wav_files = list(wav_dir.glob("*.wav"))
    print(f"Found {len(wav_files)} wav files for language '{lang}'.")

    # Initialize Phonemizer backend
    # We use 'espeak-ng' as backend, ensuring punctuation is handled
    backend = EspeakBackend(
        language='en-us', 
        preserve_punctuation=True, 
        with_stress=True
    )

    for wav_path in wav_files:
        stem = wav_path.stem
        txt_path = txt_dir / f"{stem}.txt"

        if not txt_path.exists():
            print(f"Missing text file for {wav_path.name}, skipping.")
            continue

        # Read reference text
        with open(txt_path, 'r', encoding='utf-8') as f:
            ref_text = f.read().strip()

        # Generate Phonemes
        # separator.phone=None means no separator between phonemes
        try:
            ref_phon = backend.phonemize([ref_text], strip=True)[0]
        except Exception as e:
            print(f"Error phonemizing {stem}: {e}")
            continue

        # Calculate Audio Metadata
        audio_md5 = get_audio_md5(wav_path)
        
        # Get duration and sample rate
        try:
            info = sf.info(wav_path)
            duration = info.duration
            sr = info.samplerate
        except Exception as e:
            print(f"Error reading audio {wav_path}: {e}")
            continue

        # Construct unique utterance ID: {lang}_{stem}
        utt_id = f"{lang}_{stem}"

        # Create record
        record = {
            "utt_id": utt_id,
            "lang": lang,
            "wav_path": str(wav_path),
            "ref_text": ref_text,
            "ref_phon": ref_phon,
            "sr": sr,
            "duration_s": duration,
            "audio_md5": audio_md5,
            "snr_db": None  # Clean audio has no added noise
        }
        records.append(record)

    # Output manifest path: data/manifests/en/clean.jsonl
    out_manifest = out_dir / lang / "clean.jsonl"
    write_manifest(records, out_manifest)


def main():
    # Load parameters
    with open("params.yaml", 'r') as f:
        params = yaml.safe_load(f)

    raw_data_dir = Path(params['prepare']['raw_data_dir'])
    output_dir = Path(params['prepare']['output_dir'])
    languages = params['base']['languages']

    for lang in languages:
        process_language(lang, raw_data_dir, output_dir)


if __name__ == "__main__":
    main()