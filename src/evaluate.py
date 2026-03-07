import os
import sys
import yaml
import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

def levenshtein_distance(ref, hyp):
    """
    Calculates the Levenshtein distance between two sequences (strings).
    Returns (distance, reference_length).
    """
    m = len(ref)
    n = len(hyp)
    
    # Initialize matrix
    d = np.zeros((m + 1, n + 1), dtype=int)
    
    for i in range(m + 1):
        d[i, 0] = i
    for j in range(n + 1):
        d[0, j] = j
        
    for j in range(1, n + 1):
        for i in range(1, m + 1):
            if ref[i - 1] == hyp[j - 1]:
                cost = 0
            else:
                cost = 1
            d[i, j] = min(d[i - 1, j] + 1,      # deletion
                          d[i, j - 1] + 1,      # insertion
                          d[i - 1, j - 1] + cost) # substitution
                          
    return d[m, n], m

def calculate_per(manifest_path):
    """
    Reads a manifest and computes the average PER.
    PER = (S + D + I) / N
    """
    total_distance = 0
    total_ref_len = 0
    
    with open(manifest_path, 'r', encoding='utf-8') as f:
        for line in f:
            record = json.loads(line)
            # Remove spaces to treat as sequence of characters (phonemes)
            # or split by space if phonemes are space-separated.
            # espeak output is usually space-separated words, but chars are phonemes.
            # Let's clean spaces to compare pure phoneme sequences.
            ref = record['ref_phon'].replace(" ", "")
            pred = record.get('pred_phon', "").replace(" ", "")
            
            dist, length = levenshtein_distance(ref, pred)
            total_distance += dist
            total_ref_len += length
            
    if total_ref_len == 0:
        return 0.0
        
    return total_distance / total_ref_len

def process_evaluation(lang, params):
    input_dir = Path(params['evaluate']['input_dir']) / lang
    metrics_file = Path(params['evaluate']['metrics_file'])
    plots_dir = Path(params['evaluate']['plots_dir'])
    
    plots_dir.mkdir(parents=True, exist_ok=True)
    
    # Find all jsonl files (clean + snrX)
    manifests = list(input_dir.glob("*.jsonl"))
    
    results = {}
    
    for man in manifests:
        # Determine SNR from filename
        name = man.stem # "clean" or "snr20"
        
        per = calculate_per(man)
        
        if name == "clean":
            snr_val = float('inf') # Infinite SNR for clean audio
        elif name.startswith("snr"):
            snr_val = float(name.replace("snr", ""))
        else:
            continue
            
        results[name] = {
            "snr": snr_val,
            "per": per
        }
        print(f"[{lang}] {name}: PER = {per:.2%}")

    # Save metrics to JSON (DVC can track this)
    # We organize it as nested structure for DVC metrics
    full_metrics = {lang: results}
    
    # If metrics file exists, merge (for multi-language support later)
    if metrics_file.exists():
        with open(metrics_file, 'r') as f:
            existing = json.load(f)
        existing.update(full_metrics)
        full_metrics = existing
        
    with open(metrics_file, 'w') as f:
        json.dump(full_metrics, f, indent=2)
        
    return results

def plot_results(all_results, params):
    plots_dir = Path(params['evaluate']['plots_dir'])
    output_plot = plots_dir / "per_vs_noise.png"
    
    plt.figure(figsize=(10, 6))
    
    # Data structure to hold PERs for each SNR across all languages
    # snr_map = { 20: [0.1, 0.2], 10: [0.3, 0.4] ... }
    snr_map = {}

    # 1. Plot individual languages
    for lang, data in all_results.items():
        points = []
        for name, metrics in data.items():
            snr = metrics['snr']
            per = metrics['per']
            
            # Skip 'clean' (inf) for plotting or handle separately
            if snr != float('inf'):
                points.append((snr, per))
                
                # Collect for mean calculation
                if snr not in snr_map:
                    snr_map[snr] = []
                snr_map[snr].append(per)
        
        # Sort by SNR
        points.sort(key=lambda x: x[0])
        
        if points:
            snrs, pers = zip(*points)
            plt.plot(snrs, pers, marker='o', label=f"Language: {lang}", alpha=0.6)

    # 2. Calculate and Plot Cross-language Mean
    if snr_map:
        mean_points = []
        for snr, per_list in snr_map.items():
            mean_per = sum(per_list) / len(per_list)
            mean_points.append((snr, mean_per))
        
        # Sort by SNR
        mean_points.sort(key=lambda x: x[0])
        
        mean_snrs, mean_pers = zip(*mean_points)
        # Plot mean with a thicker, distinct line (e.g., Black Dashed)
        plt.plot(mean_snrs, mean_pers, color='black', linewidth=2.5, linestyle='--', marker='s', label="Cross-Language Mean")

    plt.title("Phoneme Error Rate (PER) vs Noise Level (SNR)")
    plt.xlabel("SNR (dB)")
    plt.ylabel("PER (Lower is Better)")
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)
    plt.legend()
    
    plt.savefig(output_plot)
    print(f"Plot saved to {output_plot}")

def main():
    with open("params.yaml", 'r') as f:
        params = yaml.safe_load(f)
        
    languages = params['base']['languages']
    
    all_results = {}
    for lang in languages:
        all_results[lang] = process_evaluation(lang, params)
        
    plot_results(all_results, params)

if __name__ == "__main__":
    main()