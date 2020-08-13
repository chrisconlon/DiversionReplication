#!/usr/bin/env bash
set -e

## If you are in main directory with run_all.sh
# you will need to go to code to run everything

# Install Packages
pip install -r requirements.txt
pip install git+https://github.com/jeffgortmaker/pyblp/

# make sure you have the latest
# requires git, which may require xcode!
cd code

# Get BLP and Nevo Cases: Comment out when /dict/ is populated
python run_all_cases.py

# Table Generating Block
python tab34_params.py
python tab56_diversion.py
python tab7_wtp.py

# Figure Generating Block
python fig12_decomp.py
python fig34_late.py
