import os
import yaml
import json
import numpy as np
import soundfile as sf
from pathlib import Path
from utils import write_manifest, get_audio_md5

# provided code in lab page 3
def add_noise(signal: np.ndarray, snr_db: float, rng: np.random.Generator) -> np.ndarray:
    signal_power = np.mean(signal ** 2)

    # Avoid division by zero for silent signals
    if signal_power == 0:
        return signal

    snr_linear = 10 ** (snr_db / 10)
    noise_power = signal_power / snr_linear

    noise = rng.normal(
        loc=0.0,
        scale=np.sqrt(noise_power),
        size=signal.shape,
    )
    return signal + noise


def add_noise_to_file(input_wav: str, output_wav: str, snr_db: float, seed: int | None = None) -> None:
    """
    Reads a wav file, adds noise, and writes it back.
    """
    signal, sr = sf.read(input_wav)
    
    # if signal.ndim != 1:
    #     raise ValueError("Only mono audio is supported")
    
    if signal.ndim != 1:
        # Take first channel if stereo
        signal = signal[:, 0]

    rng = np.random.default_rng(seed)
    noisy_signal = add_noise(signal, snr_db, rng)
    
    # Write output
    sf.write(output_wav, noisy_signal, sr)


# Pipeline Stage Logic
def process_augmentation(lang, params):
    """
    Main logic to iterate over clean manifests and generate noisy versions.
    """
    input_dir = Path(params['augment']['input_manifest_dir'])
    output_manifest_dir = Path(params['augment']['output_manifest_dir'])
    output_audio_dir = Path(params['augment']['output_audio_dir'])
    
    snr_levels = params['augment']['snr_levels']
    global_seed = params['augment']['seed']

    # Path to the clean manifest from Stage 1
    clean_manifest_path = input_dir / lang / "clean.jsonl"
    
    if not clean_manifest_path.exists():
        print(f"Warning: Clean manifest not found at {clean_manifest_path}")
        return

    print(f"Loading manifest: {clean_manifest_path}")
    clean_records = []
    with open(clean_manifest_path, 'r', encoding='utf-8') as f:
        for line in f:
            clean_records.append(json.loads(line))

    # Iterate over each SNR level
    for snr in snr_levels:
        print(f"  > Generating noise for {lang} at SNR {snr}dB...")
        
        noisy_records = []
        
        # Create output directory for this specific SNR: data/audio_noisy/en/snr10/
        snr_name = f"snr{snr}"
        current_audio_dir = output_audio_dir / lang / snr_name
        current_audio_dir.mkdir(parents=True, exist_ok=True)
        
        for record in clean_records:
            utt_id = record['utt_id']
            clean_wav_path = record['wav_path']
            
            # Define new filename: {stem}_{snr}.wav
            stem = Path(clean_wav_path).stem
            noisy_filename = f"{stem}_{snr_name}.wav"
            noisy_wav_path = current_audio_dir / noisy_filename
            
            # Deterministic seed per file: global_seed + hash(utt_id) + snr
            # This ensures if we re-run just one file, it gets the same noise.
            # Using absolute(hash) to ensure positive integer.
            local_seed = (global_seed + abs(hash(utt_id)) + int(snr)) % (2**32)
            
            # Generate and save audio
            add_noise_to_file(
                input_wav=clean_wav_path,
                output_wav=str(noisy_wav_path),
                snr_db=float(snr),
                seed=local_seed
            )
            
            # Create new manifest record
            noisy_record = record.copy()
            noisy_record['wav_path'] = str(noisy_wav_path)
            noisy_record['snr_db'] = snr
            noisy_record['audio_md5'] = get_audio_md5(noisy_wav_path)
            # ref_text, ref_phon, duration_s, sr remain the same
            
            noisy_records.append(noisy_record)
        
        # Write manifest: data/manifests_noisy/en/snr10.jsonl
        out_manifest_path = output_manifest_dir / lang / f"{snr_name}.jsonl"
        write_manifest(noisy_records, out_manifest_path)


def main():
    with open("params.yaml", 'r') as f:
        params = yaml.safe_load(f)
    
    languages = params['base']['languages']
    
    for lang in languages:
        process_augmentation(lang, params)

if __name__ == "__main__":
    main()