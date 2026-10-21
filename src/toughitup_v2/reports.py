from datetime import datetime
import sys
import textwrap
from toughitup_v2 import directories
from toughitup_v2.directories import MAIN_SCRIPT_DIR



def start_report(start_text):
    print(start_text)
    with open(directories.log_file, 'w', encoding='utf-8') as f:
        f.write(start_text)

def print_in_report(line):
    print(line)
    with open(directories.log_file, "a", encoding="utf-8") as f:
        f.write(f"{line}\n")

def print_wrapped(text, width = 80):
    wrapped_lines = textwrap.fill(text, width=width)
    print_in_report(wrapped_lines)

def print_chapter_report(chapter):
    return f"{chapter:<25}" + "="*55 + "\n"


def print_keywords_report(keywords):

    print_in_report(f" - Keywords found: {len(keywords)}\n")
    print_in_report("keywords list")
    print_in_report("-"*40)

    for i in range(0, len(keywords), 8):
        print_in_report(" - ->" + " ".join(keywords[i:i+8]))
    print_in_report(f"\n")


def print_initialization():
    print_in_report(f"\n@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@ TOUGHITUP - Toughing you up! @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@")
    print_in_report(f"@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@  Starting Conversion  @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@\n")


block_heather_A = "----1----*----2----*----3----*----4----*----5----*----6----*----7----*----8"
block_heather_B = "----2----3----4----5----6----7----8----9---10---11---12---13---14---15---16"

banner = f"""

================================================================================
     TOUGHITUP   {MAIN_SCRIPT_DIR}
     Started:    {datetime.now().strftime("%d-%m-%Y %H:%M:%S")}
     Python:     {sys.version.split()[0]}
================================================================================
"""