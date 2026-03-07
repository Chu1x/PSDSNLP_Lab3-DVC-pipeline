import json
import os
import hashlib
import tempfile
import shutil
from pathlib import Path

def get_audio_md5(file_path):
    """
    Calculates the MD5 checksum of a file.
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
    output_path = Path(output_path)
    # Ensure parent directory exists
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Create a temp file in the same directory to ensure atomic move works
    # (moving across filesystems is not atomic)
    temp_dir = output_path.parent
    with tempfile.NamedTemporaryFile('w', dir=temp_dir, delete=False, encoding='utf-8', suffix='.tmp') as tmp:
        temp_name = tmp.name
        try:
            for record in data:
                # json.dump writes a JSON object; adding \n makes it JSONL
                tmp.write(json.dumps(record, ensure_ascii=False) + '\n')
        except Exception as e:
            tmp.close()
            os.remove(temp_name)
            raise e

    # Atomic rename/move: rename temp file to final file
    shutil.move(temp_name, output_path)
    print(f"Created manifest: {output_path}")