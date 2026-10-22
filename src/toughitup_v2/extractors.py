import numpy as np
import ast

from itertools import repeat
from typing import Iterable, Iterator, List, Optional

from . import reports



def read_grdecl_file(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    return lines

def parse_value(value):
    try:
        return int(value)
    except ValueError:
        try:
            return float(value)
        except ValueError:
            return value

def find_keywords(file_content):
    keywords = []
    with open(file_content, 'r') as f:
        for line in f:
            stripped = line.split('--', 1)[0].strip()
            if not stripped or stripped.startswith('--') or stripped.startswith('/'):
                continue
            parts = stripped.split()
            if len(parts) == 1 and parts[0].isupper() and parts[0].isalpha():
                keywords.append(parts[0])

    return np.array(keywords)

def extract_data_list(lines, keyword):
    extracted_data = _extract_data_tokens(lines, keyword)
    if extracted_data is None:
        return

    reports.print_in_report(f"Keyword {keyword} imported correctly...")
    extracted_list = [parse_value(value) for value in extracted_data]
    return extracted_list

def get_units(file, keyword):
        info = extract_data_list(file, keyword)
        info = info[0]
        try:
            units = ast.literal_eval(info)
        except (ValueError, SyntaxError):
            units = info
        return units

def _extract_data_tokens(lines, keyword) -> Optional[List[str]]:
    extracted_data: List[str] = []
    reading = False

    for line in lines:
        line = line.split('--')[0].strip()
        if not line:
            continue

        parts = line.split()
        if parts[0].upper() == keyword.upper():
            reading = True
            continue

        if reading:
            if '/' in line:
                line = line.split('/')[0]
                extracted_data.extend(line.split())
                break
            else:
                extracted_data.extend(parts)

    if not reading:
        print(f"Keyword '{keyword}' not found in file.")

    return extracted_data

def _iter_numeric_values(tokens: Iterable[str]) -> Iterator[float]:
    for token in tokens:
        count_text, separator, value_text = token.partition('*')
        if not separator:
            yield float(token)
            continue

        try:
            count = int(count_text)
            value = float(value_text)

        except ValueError as error:
            raise ValueError(
                f"Invalid repetition token {token!r}: expected a non-negative "
                "integer count followed by '*' and a numeric value."
            ) from error

        if count<0:
            raise ValueError( 
                f"Invalid repetition token {token!r}: count must be non-negative."
                )
        yield from repeat(value, count)

def extract_data_to_np(lines, keyword):
    extracted_data = _extract_data_tokens(lines, keyword)

    if extracted_data is None:
        return

    reports.print_in_report(f"Keyword {keyword} imported correctly.")
    return np.fromiter(
        _iter_numeric_values(extracted_data), 
        dtype=float,
    )