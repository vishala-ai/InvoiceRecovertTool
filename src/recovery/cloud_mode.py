"""Explicit deployment selection, without loading credential files."""
import os


def is_cloud():
    return os.environ.get('RECOVERY_CLOUD') == '1'
