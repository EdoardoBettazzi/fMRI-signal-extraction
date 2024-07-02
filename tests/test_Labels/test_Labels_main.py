'''Extracts bold signals from BIDS dataset, based on a csv configuration if provided, or interactively if configuration is not provided.

Takes in input:
    - path to the BIDS-formatted fmriprep dataset. Required.
    - csv containing list of ROIs (as labels integers) or list of Seeds (as xyz coordinates). Required
    - path to the parcellation atlas files. Required.
    Execute

Returns in output:
'''

from bids import BIDSLayout
import logging
import os
from pathlib import Path
from test_utils import helpers, test_checking, test_extracting
import sys

if __name__ == "__main__":

    ## User arguments
    args = helpers.test_user_args()

    ## Set up logging
    logging.basicConfig(filename=os.path.join(args.output,'config.log'),
                        filemode='w',
                        level=logging.INFO,
                        format='%(asctime)s:%(levelname)s:%(message)s')  
    
    ## Create Layout object to parse input BIDS dataset
    derivatives_path = args.input
    layout = BIDSLayout(derivatives_path, absolute_paths=True, config=['bids','derivatives'])

    ## Parse configuration file, or interact to create one
    if args.configuration:

        # Check and return verified configuration
        config_df = test_checking.config(args.configuration, layout)
        if config_df is None:
            sys.exit("Something's wrong with the configuration file. See the log for more details.")
    
    else:
        # Create configuration file interactively
        sys.exit('WORK IN PROGRESS')
    
    ## Get the functional scan
    functional_files = helpers.test_get_scans(layout, config_df[1])

    ## Create folder to store extracted signals
    output_folder = Path(os.path.join(args.output, 'bold_signals'))
    output_folder.mkdir(parents=True, exist_ok=True)
    
    ## Extract signals
    reduction_strategy = config_df[7][0]
    if reduction_strategy == 'mean':
        # For a subset of ROIs
        if args.rois:
            sys.exit('WORK IN PROGRESS')
        else:
            # For all the ROIs in the atlas
            test_extracting.test_mean(func_files=functional_files,
                                    configuration_df=config_df,
                                    atlas=args.parcels,
                                    output_folder=str(output_folder.absolute()),
                                    matlab=args.matlab
                                    )

            if args.DCM:
                helpers.dcm_inputs(configuration_df=config_df, output_folder=args.output, bold_signals_folder=str(output_folder.absolute()))

    if reduction_strategy == 'SVD':
        # For a subset of ROIs
        if args.rois:
            sys.exit('WORK IN PROGRESS')
        else:
            # For all the ROIs in the atlas
            test_extracting.test_first_eig(func_files=functional_files,
                                    configuration_df=config_df,
                                    atlas=args.parcels,
                                    output_folder=str(output_folder.absolute()),
                                    matlab=args.matlab
                                    )
            
            if args.DCM:
                helpers.dcm_inputs(configuration_df=config_df, output_folder=args.output, bold_signals_folder=str(output_folder.absolute()))


"""

    ## parse config, or interact

        if configuration csv is provided:
            perform sanity check
            reformat if necessary
            parse configuration into bids_query_dataframe and extraction_strategy_dataframe
                
        else: 
            query BIDS dataset and save query in dataframe 'bids_query_dataframe'
            customize extraction (confounds removal and reduction strategy) and save config in 'extraction_strategy_dataframe'
            merge dataframes and output csv file for later use

    ## get the functional scans

        if subject list is provided:
            get subjects functional scans
        else:
            get all functional scans

    ## extract signals

        if a csv with ROIs labels, roi-based:
            
            if parcellations are not provided:
                fetch Schaefer parcellations from nilearn

            if reduction strategy is mean:
                apply mean extraction based on roi list, or for whole brain

            else if it is first-eigenvariate:
                apply first-eigenvariate extraction based on Labels
        
        if a csv of coordinates is provided, seed-based:
        
            if reduction strategy is mean:
                apply mean extraction based on coordinates list

            else if it is first-eigenvariate:
                apply first-eigenvariate extraction based on coordinates list
        
    
"""