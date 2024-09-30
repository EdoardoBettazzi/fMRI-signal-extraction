import logging
import pandas as pd
import json
import os

def create_dataset_description(directory):
    file_path = os.path.join(directory, 'dataset_description.json')
    
    if not os.path.exists(file_path):
        template = {
            "Name": "Dataset derivatives",
            "BIDSVersion": "",
            "HEDVersion": "",
            "DatasetType": "derivative",
            "License": "",
            "Authors": ["", "", ""],
            "Acknowledgements": "",
            "HowToAcknowledge": "",
            "Funding": ["", "", ""],
            "EthicsApprovals": [""],
            "ReferencesAndLinks": ["", "", ""],
            "DatasetDOI": "doi:",
            "GeneratedBy": [{"Name": "", "Version": ""}]
        }
        
        with open(file_path, 'w') as f:
            json.dump(template, f, indent=2)
        print(f"Created dataset_description.json in {directory}")
    else:
        print(f"dataset_description.json already exists in {directory}")

def config(configuration_file, layout):

    '''
    Checks that the configuration file contains all the necessary information.
    If so, it splits the info into two dataframes: one to query the BIDS dataset, and 
    another to orient the signal extraction.

    Parameters:
        configuration_file: 'csv'
            csv file containing the necessary information to perform signal extraction. See README.MD for details.
        layout: BIDSLayout
            Layout object representing an entire BIDS dataset.
    
    Returns:
        config_df: 'Pandas DataFrame Obj', or None
            Dataframe containing the verified information to perform the signal extraction.
            If the verification fails, None is returned.
    '''
    
    # Check if the configuration file exists
    try:
        # Read the file as dataframe
        config_df = pd.read_csv(configuration_file, header=None, keep_default_na=False, na_values=['NaN'])
        
        # Check that all 4 columns are present
        cols = config_df.shape[1]
        if cols < 12:
            print('One or more necessary columns in the configuration file are missing.')
            logging.error(f'Missing columns. Should be 4, but it is {cols}')
            return None

        # Check BIDS dataset querying parameters (1+2 columns)
        if BIDSquery(config_df, layout):
            pass
        else:
            print('One or more values for querying the BIDS dataset are not available or incorrect. Please, check the configuration file and try again.')
            logging.error('One or more values for querying the BIDS dataset are incorrect.')
            return None
            
        # Check signal extraction parameters (3+4 columns)
        _config_df_confounds = confounds(config_df)
        if _config_df_confounds is not None:
            config_df = _config_df_confounds
        else:
            return None
        
        # Check smoothing and temporal filtering (5+6 columns)
        _config_df_smoothing = smoothing(config_df)
        if _config_df_smoothing is not None:
            config_df = _config_df_smoothing
        else:
            return None
        
        _config_df_filter = temp_filters(config_df)
        if _config_df_filter is not None:
            config_df = _config_df_filter
        else:
            return None
        
        # Check reduction strategy (7+8 columns)
        reduction_strategy = config_df[7][0]
        if reduction_strategy.upper() in ['MEAN', 'SVD']:
            pass
        else:
            print(f'Reduction strategy {reduction_strategy} is not available.\
            Please, check README.MD for the available strategies.')
            logging.error(f'Reduction strategy {reduction_strategy} is not available. Please, check README.MD for the available strategies.')
            return None
        
        # Check ROIs labels and names (9+10 and 11+12 columns)
        if rois(config_df):
            pass
        else:
            return None
                       
        # Everything in the right place: return the configuration dataframe
        return config_df
           
    except FileNotFoundError:
        print("Can't find the configuration csv file. Are you sure the path/to/the/file is correct?")
        logging.error(FileNotFoundError)
    
    return None


def BIDSquery(configuration_df, layout):

    '''
    Parameters:
        configuration_df: 'Pandas DataFrame Obj'
            dataframe containing in 2nd columns the values of the parameters 
            that are necessary to query the BIDS dataset.
        layout: BIDSLayout
            Layout object representing an entire BIDS dataset.
    
    Returns:
        True, if the values are well put; False, otherwise.
    '''
    
    dtype = True if configuration_df[1][0] in layout.get_datatype() else False
    task = True if configuration_df[1][1] in layout.get_task() else False
    preprocessing = True if configuration_df[1][2] in ['preproc', 'smoothAROMAnonaggr'] else False
    session = True if configuration_df[1][3] in layout.get_session() or configuration_df[1][3] == 'None' else False
    if configuration_df[1][3] == 'None':
        configuration_df.iloc[3, 1] = None
    template = True if configuration_df[1][4] in ['MNI152NLin2009cAsym', 'MNI152NLin6Asym'] else False
    
    try:
        if dtype and task and preprocessing and session and template:
            # Check that template matches preprocessing
            if configuration_df[1][2] == 'smoothAROMAnonaggr' and configuration_df[1][4] == 'MNI152NLin6Asym':
                return True
            elif configuration_df[1][2] == 'preproc' and configuration_df[1][4] == 'MNI152NLin2009cAsym':
                return True
            else:
                print(f'Template {configuration_df[1][4]} is not matching the preprocessing procedure {preprocessing} \
                in the configuration file.')
                return False
        else:
            raise ValueError
        
    except ValueError:
        
        if not dtype:
            logging.error('Datatype value is incorrect')
        if not task:
            logging.error('Task value is incorrect')
        if not preprocessing:
            logging.error('Preprocessing value is incorrect')
        if not session:
            logging.error('Session value is incorrect')
        if not template:
            logging.error('Template value is incorrect')

        return False


def rois(configuration_df):

    '''
    Parameters:
        configuration_df: 'Pandas DataFrame Obj'
            dataframe containing in 9th and 11th columns the ROis labels/coordinates and anatomical names.
        layout: BIDSLayout
            Layout object representing an entire BIDS dataset.
    
    Returns:
        True, if the values are well put; False, otherwise.
    '''
    
    try:
        labels_coords = (configuration_df[9][:]).tolist()
        if len(labels_coords) == 0:
            raise ValueError

        names = (configuration_df[11][:]).tolist() 
        if len(names) == 0:
            raise ValueError
        
        # Number of labels not matching number of names
        if len(labels_coords) != len(names):
            raise ValueError
        
        return True

    except ValueError:
        logging.error('Something is wrong with the specification of ROIs. Be sure to specify both fields, and that the columns match in length.')
        return False


def confounds(configuration_df):
    
    '''
    Parameters:
        configuration_df: 'Pandas DataFrame Obj'
            Dataframe containing in 4th column the values of the parameters 
            that are necessary to regress out the confounds.
            
    Returns:
        configuration_df: 'Pandas DataFrame Obj', or False
            Same as the input dataframe, with numerical or 'None' strings replaced
            with int/float or None values.
            If any check is failed, False is returned. 
    '''
        
    denoise_strategy = configuration_df[3][0]
    preset_strategies = ['simple', 'scrubbing', 'compcor', 'ica_aroma']
    
    # Check denoise strategy
    if denoise_strategy in preset_strategies:
        try:
            if denoise_strategy == preset_strategies[0]:
                # Conditions for motion, wm_csf, global_signal
                motion = configuration_df[3][1]
                wm_csf = configuration_df[3][2]
                global_signal = configuration_df[3][3]
                if global_signal == 'None':
                    global_signal = None
                    configuration_df.iloc[3,3] = global_signal


                condition1 = True if all(value in ['basic', 'power2', 'derivatives', 'full'] for value in [motion, wm_csf]) else False
                condition2 = True if global_signal in ['basic', 'power2', 'derivatives', 'full', None] else False

            elif denoise_strategy == preset_strategies[1]:
                # Conditions for motion, wm_csf, global_signal, scrub, fd_thr, std_var_thr
                motion = configuration_df[3][1]
                global_signal = configuration_df[3][2]
                if global_signal == 'None':
                    global_signal = None
                    configuration_df.iloc[2,3] = global_signal  
                scrub = float(configuration_df[3][3])
                fd_threshold = float(configuration_df[3][4])
                std_dvars_threshold = float(configuration_df[3][5])
                if scrub < 0 or fd_threshold < 0 or std_dvars_threshold < 0:
                    raise ValueError
                else:
                    # Reassign variables in the array
                    configuration_df[3][3] = scrub
                    configuration_df[3][4] = fd_threshold
                    configuration_df[3][5] = std_dvars_threshold

                condition1 = True if motion in ['basic', 'power2', 'derivatives', 'full'] else False
                condition2 = True if global_signal in ['basic', 'power2', 'derivatives', 'full', None] else False

            elif denoise_strategy == preset_strategies[2]:
                # Conditions for motion, compcor, n_compcor
                motion = configuration_df[3][1]
                compcor = configuration_df[3][2]
                n_compcor = configuration_df[3][3]
                if n_compcor == 'all':
                    pass
                elif n_compcor.isdigit():
                    configuration_df.iloc[3,3] = int(n_compcor)
                else:
                    raise ValueError

                condition1 = True if motion in ['basic', 'power2', 'derivatives', 'full'] else False
                condition2 = True if compcor in ['anat_combined', 'anat_separated', 'temporal', 'temporal_anat_combined', 'temporal_anat_separated'] else False

            elif denoise_strategy == preset_strategies[3]:
                # Conditions for wm_csf, global_signal
                wm_csf = configuration_df[3][1]
                global_signal = configuration_df[3][2]
                if global_signal == 'None':
                    global_signal = None
                    configuration_df.iloc[2,3] = global_signal
                    
                condition1 = True if wm_csf in ['basic', 'power2', 'derivatives', 'full'] else False
                condition2 = True if global_signal in ['basic', 'power2', 'derivatives', 'full', None] else False

            # Check conditions
            if condition1 and condition2:
                # Return the corrected df
                return configuration_df
            else:
                raise ValueError

        except ValueError:
                print(f"One or more values for the {denoise_strategy} denoise strategy parameters are not available or incorrect. Please, check the configuration file and try again.")
                logging.error(f'One or more values for the {denoise_strategy} denoise strategy parameters are not available or incorrect.')

                return None

    else:
        print(f'Denoising strategy "{denoise_strategy}" is not available. Please, check the configuration file and choose one among {preset_strategies}')
        logging.error(f'The denoising strategy {denoise_strategy} is not available.')
        return None
    

def smoothing(configuration_df):
    
    '''
    Parameters:
        configuration_df: 'Pandas DataFrame Obj'
            dataframe containing in 6th column the value of the smoothing FWHM parameter.
            
    Returns:
        configuration_df: 'Pandas DataFrame Obj'
            Same as the input dataframe, with numerical or 'None' strings replaced
            with int/float or None values.
    '''

    try:
        smoothing_fwhm = configuration_df[5][0]
        if smoothing_fwhm == 'None':
            configuration_df.iloc[0,5] = None
        elif smoothing_fwhm == 'fast':
            pass
        elif float(smoothing_fwhm) > 0:
            configuration_df.iloc[0,5] = float(smoothing_fwhm)
        else:
            raise ValueError
            
        return configuration_df
    
    except ValueError:
        print('Something is wrong with the smoothing_fwhm parameter.\
        Please, check the third column of the configuration file, and try again.')
        logging.error('Input value for smoothing_fwhm is incorrect.')
        return None
    

def temp_filters(configuration_df):
    
    '''
    Parameters:
        configuration_df: 'Pandas DataFrame Obj'
            dataframe containing in 6th column the values of the temporal filtering parameters.
            
    Returns:
        configuration_df: 'Pandas DataFrame Obj'
            Same as the input dataframe, with numerical or 'None' strings replaced
            with int/float or None values.
    ''' 
    
    try:
        high_pass = configuration_df[5][1]
        if high_pass == 'None':
            configuration_df.iloc[1,5] = None
        elif float(high_pass) > 0:
            configuration_df.iloc[1,5] = float(high_pass)
        else:
            raise ValueError

        low_pass = configuration_df[5][2]
        if low_pass == 'None':
            configuration_df.iloc[2,5]  = None
        elif float(low_pass) > 0:
            configuration_df.iloc[2,5] = float(low_pass)
        else:
            raise ValueError
        
        return configuration_df
        
    except ValueError:
        print('Something is wrong with high_pass and low_pass parameters.\
        Please, check the third column of the configuration file, and try again.')
        logging.error('Input values for high_pass and low_pass are incorrect.')
        return None