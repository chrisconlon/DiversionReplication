# Empirical Properties of Diversion Ratios
Conlon and Mortimer (2020)
A copy of the paper is here: https://chrisconlon.github.io/site/diversion.pdf

## Before running code
To download the repo simply type:

    git clone https://github.com/chrisconlon/DiversionReplication

### Dataset Size and Memory
1. Total runtime on a 2015 iMac with 64GB of RAM is around 40 minutes. 
Runtime on a 2017 MacBook Air with iMac with 8GB of RAM is around three hours.
2. All of the datasets saved should take up less than 120 MB of drive space.
3. The file run_all_cases.py need only be run once to populate data/dict. The file takes about 2.5 hours to run.

## How to run code
Change to the directory containing this file and run "bash run_all.sh" on the terminal. The code should take approximately 1-2 hours to run in its entirety. Tables and figures will be produced as described below.

Windows Users: instead use "run_all.bat" from the command prompt.

## File of origin for tables and figures

| Table / Figure Number                 | Generating File           | File Name                    |
| ---                                   | ---                       | ---                          |
| Table 1                               | (by hand)                 | N/A                          |
| Table 2                               | (by hand)                 | N/A                          |
| Table 3                               | tab34_params.py           | tab3_params.tex              |
| Table 4                               | tab34_params.py           | tab4_stats.tex               |
| Table 5                               | tab56_diversion.py        | tab5_div.tex                 |
| Table 6                               | tab56_diversion.py        | tab6_rel.tex                 |
| Table 7                               | tab7_wtp.py               | tab7_wtp.tex                 |
| Table B1, Panel A                     | tab34_params.py           | tableB1_blp.tex              |
| Table B1, Panel B                     | tab34_params.py           | tableB1_nevo.tex             |


## Within-File Dependencies:

run_all_cases.py
    
    aux_nevo_cases
    aux_blp_cases

tab34_params.py
     
    aux_table_functions / load_pyblp_dict, get_params_nevo, get_params_blp, outreg


tab56_diversion.py: 

    aux_table_functions / load_pyblp_dict

tab7_wtp.py: 

    aux_table_functions / load_blp_base, do_single_market, do_single_market_indiv, reshape_wtp
    aux_plot_config
    aux_nevo_cases / get_nevo_base

fig12_decomp.py: 

    aux_plot_config
    aux_double_plot / plot_double_mpc
    aux_table_functions / draw_blp_agents, load_blp_base

fig34_late.py: 

    aux_plot_config
    aux_late / plot_late_single
    aux_table_functions / draw_blp_agents, load_blp_base

## Python  dependencies
Python (version 3.4 or above) - install dependencies with 

    pip3 install -r requirements.txt

    numpy, pandas, matplotlib, scipy, tabulate, pyarrow, brotli

Note: Windows 10 has trouble with older versions of Python 3 and multiprocessing (unrelated to this project). Please use 3.7.7 or above.

## Files Provided

data/raw:

1. blp_product_data_opt.parquet: BLP auto data corrected using pre-computed optimal instruments. See Conlon and Gortmaker (2020) for a detailed discussion.
2. nevo_product_data_opt.parquet: Nevo (fake) cereal data corrected using pre-computed optimal instruments. See Conlon and Gortmaker (2020) for a detailed discussion.
