import numpy as np


def read_grdecl_file(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    return lines

def find_keywords(file_name):
    keywords = []
    with open(file_name, 'r') as f:
        for line in f:
            stripped = line.strip()
            if not stripped or stripped.startswith('--') or stripped.startswith('/'):
                continue
            parts = stripped.split()
            if len(parts) == 1 and parts[0].isuper() and parts[0].isalpha():
                keywords.append(parts[0])

    return np.array(keywords)