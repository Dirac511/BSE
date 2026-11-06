#!/bin/bash
module load anaconda/4
PYTHON_SCRIPT='./main.py'
OUTPUT_FILE='./out_SC'
##OUTPUT_FILE='./out_mag_pi0'
python "$PYTHON_SCRIPT" > "$OUTPUT_FILE" 2>&1
echo "Over"

