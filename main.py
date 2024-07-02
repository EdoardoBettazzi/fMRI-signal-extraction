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
    layout = BIDSLayout(derivatives_path, absolute_paths=True, config=['bids','derivatives'])

    ## Parse configuration file, or interact to create one
    if args.configuration:
        # Check and return verified configuration
        config_df = checking.config(args.configuration, layout)
        if config_df is None:
            sys.exit("Something's wrong with the configuration file. See the log for more details.")
    
    else:
        sys.exit('A configuration file is needed to extract the signals.')
    
    ## Get the functional scan
    functional_files = helpers.get_scans(layout, config_df[1])

    ## Create folder to store extracted signals
    output_folder = Path(os.path.join(args.output, 'bold_signals'))
    output_folder.mkdir(parents=True, exist_ok=True)
    
    ## Extract signals
    reduction_strategy = config_df[7][0]
    if reduction_strategy == 'mean':
        extracting.mean(func_files=functional_files,
                                configuration_df=config_df,
                                atlas=args.parcels,
                                output_folder=str(output_folder.absolute()),
                                matlab=args.matlab,
                                csv=args.csv
                                )

        if args.DCM:
            helpers.dcm_inputs(configuration_df=config_df, output_folder=args.output, bold_signals_folder=str(output_folder.absolute()))

    if reduction_strategy == 'SVD':
        extracting.first_eig(func_files=functional_files,
                                configuration_df=config_df,
                                atlas=args.parcels,
                                output_folder=str(output_folder.absolute()),
                                matlab=args.matlab,
                                csv=args.csv
                                )

        if args.DCM:
            helpers.dcm_inputs(configuration_df=config_df, output_folder=args.output, bold_signals_folder=str(output_folder.absolute()))