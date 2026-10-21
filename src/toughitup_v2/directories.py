import sys
import os


def set_output_directory(directory):
    global MAIN_SCRIPT_DIR, log_file, eleme_file, conne_file, rocks_file
    global incon_file, incon_file_alt, cell_ref_csv_file, cell_ref_txt_file

    MAIN_SCRIPT_DIR = os.path.abspath(directory)
    log_file = os.path.join(MAIN_SCRIPT_DIR, "coversion_report.txt")
    # eleme_file = os.path.join(MAIN_SCRIPT_DIR, "eleme.dat")
    # conne_file = os.path.join(MAIN_SCRIPT_DIR, "conne.dat")
    # rocks_file = os.path.join(MAIN_SCRIPT_DIR, "rocks.dat")
    # incon_file = os.path.join(MAIN_SCRIPT_DIR, "incon.dat")
    # incon_file_alt = os.path.join(MAIN_SCRIPT_DIR, "incon_free_format.dat")
    # cell_ref_csv_file = os.path.join(MAIN_SCRIPT_DIR, "cell_reference.csv")
    # cell_ref_txt_file = os.path.join(MAIN_SCRIPT_DIR, "cell_reference.txt")
