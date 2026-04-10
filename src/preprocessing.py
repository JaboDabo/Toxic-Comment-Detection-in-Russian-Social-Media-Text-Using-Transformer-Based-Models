"""
Preprocessing utilities for KazSAnDRA dataset.
Note: KazSAnDRA provides a pre-cleaned `text_cleaned` column,
so this module provides only minimal additional utilities.
"""

import re


def normalize_text(text):
    """Minimal normalization: lowercase, normalize whitespace."""
    if not isinstance(text, str):
        return ""
    text = text.lower().strip()
    text = re.sub(r"\s+", " ", text)
    return text
