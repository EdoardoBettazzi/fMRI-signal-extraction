from bids import BIDSLayout
import logging
import os
import sys
from pathlib import Path
from _utils import helpers, checking, extracting

if __name__ == "__main__":

    ## User arguments
    args = helpers.user_args()

    ## Set up logging
    logging.basicConfig(filename=os.path.join(args.output,'config.log'),
                        filemode='w',
                        level=logging.ERROR,
                        format='%(asctime)s:%(levelname)s:%(message)s')  
    
    ## Create Layout object to parse input BIDS dataset
    derivatives_path = args.input
    checking.create_dataset_description(derivatives_path)
    layout = BIDSLayout(derivatives_path,
                        absolute_paths=True, 
                        config=['bids','derivatives'])

    ## Parse configuration file
    if args.configuration:
        # Check and return verified configuration
        config_df = checking.config(args.configuration, layout)
        if config_df is None:
            sys.exit("Something's wrong with the configuration file. See the log for more details.")
    
    else:
        sys.exit('A configuration file is needed to extract the signals.')
    
    ## Get the functional scan
    functional_files = helpers.get_scans(layout, config_df[1])
    
    ## Extract signals
    reduction_strategy = config_df[7][0]
    if reduction_strategy == 'mean':
        extracting.mean(func_files=functional_files,
                                configuration_df=config_df,
                                atlas=args.parcels,
                                derivatives_folder = derivatives_path,
                                output_folder=str(Path(args.output)), #.absolute()
                                matlab=args.matlab,
                                csv=args.csv,
                                DCM=args.DCM
                                )

    if reduction_strategy == 'SVD':
        extracting.first_eig(func_files=functional_files,
                                configuration_df=config_df,
                                atlas=args.parcels,
                                derivatives_folder = derivatives_path,
                                output_folder=str(Path(args.output)), #.absolute()
                                matlab=args.matlab,
                                csv=args.csv,
                                DCM=args.DCM
                                )
