# Copyright (c) 2026 Kutay Demir.
# Licensed under the PolyForm Shield License 1.0.0. See LICENSE file for details.

"""
Model Downloader Stub.

Legacy Vosk downloader has been removed as Faster-Whisper handles model management automatically.
"""

def check_model() -> bool:
    """Always return True as Whisper manages itself."""
    return True

def get_model_path() -> str:
    """Return default Whisper model name."""
    return "base.en"

if __name__ == "__main__":
    print("Faster-Whisper manages models automatically.")
