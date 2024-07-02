import argparse
import logging
import pandas as pd
from tqdm.auto import tqdm

###### NEEDS TESTING 

def test_query_BIDSdataset(layout):

    '''
    Parameters:
        layout: BIDSLayout
            Layout object representing an entire BIDS dataset.

    Returns:
        BIDSquery_df: pandas 'DataFrame Obj'                
            Dataframe collecting the information necessary to retrieve 
            the fMRI scans from the dataset.
    '''

    # Select datatype
    datatypes = layout.get_datatype()
    while True:
        try:
            dtype = input(f'Select datatype among {datatypes}: ')
            if dtype not in datatypes:
                raise ValueError
            break
        except ValueError:
            print("\nNot available.\n")

    # Select task
    tasks_list = layout.get_task()
    while True:
        try:
            task = input(f'Select task among {tasks_list}: ')
            if task not in tasks_list:
                raise ValueError
            break
        except ValueError:
            print("\nNot available.\n")

    # Select session
    sessions = layout.get_session()
    if len(sessions) > 0:
        while True:
            try:
                ses = input(f'Select session among {sessions}: ')
                if ses not in sessions:
                    raise ValueError
                break
            except ValueError:
                print("\nNot available.\n")

    # Select fMRIprep preprocessing procedure
    desc_list = ['preproc', 'smoothAROMAnonaggr']
    while True:
        try:
            desc = input(f'Select preprocessing procedure among {desc_list}: ')
            if desc not in desc_list:
                raise ValueError
            break
        except ValueError:
            print("\nNot available.\n")

    # Define brain template
    if desc == 'preproc':
        space = 'MNI152NLin2009cAsym'
    elif desc == 'smoothAROMAnonaggr':
        space = 'MNI152NLin6Asym'

    # Create pandas DataFrame
    BIDSquery_df = pd.DataFrame(data = [dtype ,task, desc, ses, space], columns=['datatype' , 'task' ,'preprocessing', 'session', 'template'])

    # Log
    logging.info(f'Dataset query details:\n {BIDSquery_df}\n')

    return BIDSquery_df

def test_customize_confounds(strategy):

    '''
    Parameters:
        strategy                string indicating denoising strategy among {simple, scrubbing, compcor, ica_aroma}, 
                                    for the default options of each preset, check nilearn.interfaces.fmriprep.load_custom_confounds_strategy
        
    Returns:
        custom_confounds                  dictionary containing the value of the parameters for a given Nilearn preset denoising strategy
    '''

    # Available preset strategies
    strategies = ['simple', 'scrubbing', 'compcor', 'ica_aroma']

    # Choose parameters settings according to the strategy
    if strategy == strategies[0]:

        while True:
            try:
                # custom_confoundsure and check
                motion = str(input('Motion, choose among {basic, power2, derivatives, full}: '))
                wm_csf = str(input('WM and CSF, choose among {basic, power2, derivatives, full}: '))
                condition1 = True if all(value in ['basic', 'power2', 'derivatives', 'full'] for value in [motion, wm_csf]) else False
                if condition1 == False:
                    raise ValueError
                    
                # custom_confoundsure and check
                global_signal = str(input('Global signal, choose among {basic, power2, derivatives, full, None}: '))
                if global_signal == 'None':
                    global_signal = None
                condition2 = True if global_signal in ['basic', 'power2', 'derivatives', 'full', None] else False
                if condition2 == False:
                    raise ValueError

                break
            except ValueError:
                print("\nNot valid.\n")
        
        # Save customization in a dataframe
        custom_confounds = pd.DataFrame(data=[strategy, motion, wm_csf, global_signal])

    elif strategy == strategies[1]:

        while True:
            try:
                motion = str(input('Motion, choose among {basic, power2, derivatives, full}: '))
                condition1 = True if motion in ['basic', 'power2', 'derivatives', 'full'] else False
                if condition1 == False:
                    raise ValueError

                global_signal = str(input('Global signal, choose among {basic, power2, derivatives, full, None}: '))
                if global_signal == 'None':
                    global_signal = None
                condition2 = True if global_signal in ['basic', 'power2', 'derivatives', 'full', None] else False
                if condition2 == False:
                    raise ValueError

                scrub = float(input('scrub, insert value (e.g 5): '))
                fd_threshold = float(input('fd_threshold, insert value (e.g 0.2): '))
                std_dvars_threshold = float(input('std_dvars_threshold, insert value (e.g 3): '))

                break
            except ValueError:
                print("\nNot valid.\n")

        # Save customization in a dataframe
        custom_confounds = pd.DataFrame(data=[strategy, motion, global_signal, scrub, fd_threshold, std_dvars_threshold])

    elif strategy == strategies[2]:

        while True:
            try:
                motion = str(input('Motion, choose among {basic, power2, derivatives, full}: '))
                condition1 = True if motion in ['basic', 'power2', 'derivatives', 'full'] else False
                if condition1 == False:
                    raise ValueError

                compcor = str(input('compcor, choose among {anat_combined, anat_separated, temporal, temporal_anat_combined, temporal_anat_separated}: '))
                condition2 = True if compcor in ['anat_combined', 'anat_separated', 'temporal', 'temporal_anat_combined', 'temporal_anat_separated'] else False
                if condition2 == False:
                    raise ValueError               

                n_compcor = input('n_compcor, type all, or insert int value: ')
                if n_compcor == 'all':
                    pass
                elif n_compcor.isdigit():
                    n_compcor=int(n_compcor)
                else:
                    raise ValueError

                break
            except ValueError:
                print("\nNot valid.\n")

        # Save customization in a dataframe
        custom_confounds = pd.DataFrame(data=[strategy, motion, compcor, n_compcor])

    elif strategy == strategies[3]:

        while True:
            try:
                wm_csf = str(input('WM and CSF, choose among {basic, power2, derivatives, full}: '))
                condition1 = True if wm_csf in ['basic', 'power2', 'derivatives', 'full'] else False
                if condition1 == False:
                    raise ValueError

                global_signal = str(input('Global signal, choose among {basic, power2, derivatives, full, None}: '))
                if global_signal == 'None':
                    global_signal = None
                condition2 = True if global_signal in ['basic', 'power2', 'derivatives', 'full', None] else False
                if condition2 == False:
                    raise ValueError
 
                break
            except ValueError:
                print('\nNot valid.\n')

        # Save customization in a dataframe
        custom_confounds = pd.DataFrame(data=[strategy, wm_csf, global_signal])
    
        
    return custom_confounds

## prompt for smoothing and filtering

## prompt for reduction strategy

def df_to_csv(*dfs):
    '''
    Takes any number of Pandas DataFrames, merges them and output in csv format.
    '''
    
    # Merge if more than 1 df
    if len(dfs) == 1:
        dfs[0].to_csv('out.csv',index=False,header=False)
    elif len(dfs) > 1:
        merged = pd.concat([dfs[0], dfs[1]], axis=1,ignore_index=False)
    
    # Output in csv format     
    merged.to_csv('test_configuration.csv', index=False, header=False)
    
    return merged


def test_get_scans(layout, BIDSquery_df, subjects_list=None):

    '''
    Parameters:
        layout: 'BIDSLayout Obj'
            Layout object representing an entire BIDS dataset.
        BIDSquery_df: 'Pandas DataFrame Obj'                
            Dataframe collecting the information necessary to retrieve the scans. 
            The first column MUST present values related to the following ['datatype' , 'task' ,'preprocessing', 'session', 'template']
            in this same exact order.
        subjects_list: 'csv'              
            List of subjects to be analysed. If passed, it retrieves the scans only for the given set of subjects

    Returns:
        scans: 'list'
            List of paths to the selected scans, either for all subjects or a subset.
    '''

    # Read DataFrame
    dtype = BIDSquery_df[0]
    task = BIDSquery_df[1]
    desc = BIDSquery_df[2]
    ses = BIDSquery_df[3]
    template = BIDSquery_df[4]

    # Get path to images
    scans = layout.get(datatype=dtype,
                        task=task,
                        desc=desc,
                        session=ses,
                        space=template,
                        extension='nii.gz',
                        return_type='file')
    
    # Manage errors and suggest possible cause
    if scans == []:
        print('Failed to retrieve the files. If you gave in input a csv configuration file, \
              please check the query values, and their order.')
        logging.error('Failed to retrieve the files, check the query values, and their order.')

    return scans
    
### add subject list selection
''''

    # Parse subjects list, if provided
    all_subjects = layout.get_subjects()
    scans = []
    logging_scans = []
    if not subjects_list:
        # Include all subjects
        scans = func_files
        pass
    else:  
        with open(args.subjects, 'r') as f:
            subs_df = pd.read_csv(f, header=None)
            subs = list(subs_df.iloc[:,0])
            for sub in subs:
                # Check if BIDS formatted
                if sub.startswith('sub-'):
                    sub = sub.partition('sub-')[2]
                if sub in all_subjects:
                    # Find the relative functional scan
                    for filepath in func_files:
                        if filepath.split('/sub-')[-1].partition('_')[0] == sub:
                            scans.append(filepath)     
                            logging_scans.append(filepath.split('/sub-')[-1].partition('_')[0])       
                else:
                    logging.error(f'Subject {sub} is not in the list of subjects, or has no functional data.')

    print(f'{len(func_files_set)} subjects under analysis.')  

    # Log subjects  
    logging.info(f'Subjects under analysis:\n {logging_scans}\n')

'''


###### TESTED AND WORKING 

def test_user_args():

    # Arguments
    parser = argparse.ArgumentParser(description='Takes in input fMRIprep preprocessed data and extracts \
                                                    bold signals (mean values or first eigenvariate) of a chosen  \
                                                    number of subjects in a ROI-based or Seed-based manner. \
                                                    The output is given in a .txt (optionally .mat) file.')

    parser.add_argument("-i", "--input", help="Path to the fMRIprep derivatives directory, the dataset must be in BIDS-format \
                                                and contain a dataset_description.json file", required=True, metavar='')
    
    parser.add_argument("-s", "--subjects", help="Optional. csv file to specify the numbers of the subjects from which you want to extract timeseries. \
                                                    The labels MUST be in the first column. \
                                                    If not provided, timeseries are extracted for all the subjects.", metavar='')

    parser.add_argument("-config", "--configuration", help="Optional. csv file to specify a how to query the BIDS dataset, \
                                                            and how to perform the signal extraction. See README.MD for details.", metavar='' )

    parser.add_argument("-p", "--parcels", help="Specify a ROI parcellation atlas file.nii.gz", required=True, metavar='')
    
    parser.add_argument("-r", "--rois", help="Optional. csv file containing the labels of the ROIs from which you want to extract timeseries. \
                                                The labels MUST be in the first column. \
                                                If not provided, timeseries are extracted from all ROIs in the atlas")
    
    parser.add_argument("-o", "--output", help="Path to the output directory where to save the extracted timeseries", required=True, metavar='')

    parser.add_argument("-mat", "--matlab", help="Optional. Save timeseries in .mat format", action="store_true", default=False)

    args = parser.parse_args()

    return args


######### CHECKING

def test_check_config(configuration_file, layout):

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
        config_df = pd.read_csv(configuration_file, header=None)
        
        # Check that all 4 columns are present
        cols = config_df.shape[1]
        if cols != 4:
            print('One or more necessary columns in the configuration file are missing.')
            logging.error(f'Missing columns. Should be 4, but it is {cols}')
            return None

        # Check BIDS dataset querying parameters (first column)
        if test_check_BIDSquery(config_df, layout):
            pass
        else:
            print('One or more values for querying the BIDS dataset are not available or incorrect. Please, check the configuration file and try again.')
            logging.error('One or more values for querying the BIDS dataset are incorrect.')
            return None
            
        # Check signal extraction parameters (second column)
        _config_df_confounds = test_check_confounds(config_df)
        if _config_df_confounds is not None:
            config_df = _config_df_confounds
        else:
            return None
        
        # Check smoothing and temporal filtering (third column)
        _config_df_smoothing = test_check_smoothing(config_df)
        if _config_df_smoothing is not None:
            config_df = _config_df_smoothing
        else:
            return None
        
        _config_df_filter = test_check_filters(config_df)
        if _config_df_filter is not None:
            config_df = _config_df_filter
        else:
            return None
        
        # Check reduction strategy (fourth column)
        reduction_strategy = config_df[3][0]
        if reduction_strategy in ['mean', 'SVD']:
            pass
        else:
            print(f'Reduction strategy {reduction_strategy} is not available.\
            Please, check README.MD for the available strategies.')
            logging.error(f'Reduction strategy {reduction_strategy} is not available. Please, check README.MD for the available strategies.')
            return None
                       
        # Everything in the right place: return the configuration dataframe
        return config_df
           
    except FileNotFoundError:
        print("Can't find the configuration csv file. Are you sure the path/to/the/file is correct?")
        logging.error(FileNotFoundError)
    
    return None


def test_check_BIDSquery(configuration_df, layout):

    '''
    Parameters:
        configuration_df: 'Pandas DataFrame Obj'
            dataframe containing the values to the parameters that are necessary
            to query the BIDS dataset.
        layout: BIDSLayout
            Layout object representing an entire BIDS dataset.
    
    Returns:
        True, if the values are well put; False, otherwise.
    '''
    
    dtype = True if configuration_df[0][0] in layout.get_datatype() else False
    task = True if configuration_df[0][1] in layout.get_task() else False
    preprocessing = True if configuration_df[0][2] in ['preproc', 'smoothAROMAnonaggr'] else False
    session = True if configuration_df[0][3] in layout.get_session() else False
    template = True if configuration_df[0][4] in ['MNI152NLin2009cAsym', 'MNI152NLin6Asym'] else False
    
    try:
        if dtype and task and preprocessing and session and template:
            # Check that template matches preprocessing
            if configuration_df[0][2] == 'smoothAROMAnonaggr' and configuration_df[0][4] == 'MNI152NLin6Asym':
                return True
            elif configuration_df[0][2] == 'preproc' and configuration_df[0][4] == 'MNI152NLin2009cAsym':
                return True
            else:
                print(f'Template {configuration_df[0][4]} is not matching the preprocessing procedure {preprocessing} \
                in the configuration file.')
                return False
        else:
            raise ValueError
    except ValueError:
            logging.error('One or more values for querying the BIDS dataset are incorrect.')
            return False
    

def test_check_confounds(configuration_df):
    
    '''
    Parameters:
        configuration_df: 'Pandas DataFrame Obj'
            Dataframe containing the values to the parameters that are necessary
            to regress out the confounds.
            
    Returns:
        configuration_df: 'Pandas DataFrame Obj', or False
            Same as the input dataframe, with numerical or 'None' strings replaced
            with int/float or None values.
            If any check is failed, False is returned. 
    '''
        
    denoise_strategy = configuration_df[1][0]
    preset_strategies = ['simple', 'scrubbing', 'compcor', 'ica_aroma']
    
    # Check denoise strategy
    if denoise_strategy in preset_strategies:
        try:
            if denoise_strategy == preset_strategies[0]:
                # Conditions for motion, wm_csf, global_signal
                motion = configuration_df[1][1]
                wm_csf = configuration_df[1][2]
                global_signal = configuration_df[1][3]
                if global_signal == 'None':
                    global_signal = None
                    configuration_df[1][3] = global_signal


                condition1 = True if all(value in ['basic', 'power2', 'derivatives', 'full'] for value in [motion, wm_csf]) else False
                condition2 = True if global_signal in ['basic', 'power2', 'derivatives', 'full', None] else False

            elif denoise_strategy == preset_strategies[1]:
                # Conditions for motion, wm_csf, global_signal, scrub, fd_thr, std_var_thr
                motion = configuration_df[1][1]
                global_signal = configuration_df[1][2]
                if global_signal == 'None':
                    global_signal = None
                    configuration_df[1][2] = global_signal  
                scrub = float(configuration_df[1][3])
                fd_threshold = float(configuration_df[1][4])
                std_dvars_threshold = float(configuration_df[1][5])
                if scrub < 0 or fd_threshold < 0 or std_dvars_threshold < 0:
                    raise ValueError
                else:
                    # Reassign variables in the array
                    configuration_df[1][3] = scrub
                    configuration_df[1][4] = fd_threshold
                    configuration_df[1][5] = std_dvars_threshold

                condition1 = True if motion in ['basic', 'power2', 'derivatives', 'full'] else False
                condition2 = True if global_signal in ['basic', 'power2', 'derivatives', 'full', None] else False

            elif denoise_strategy == preset_strategies[2]:
                # Conditions for motion, compcor, n_compcor
                motion = configuration_df[1][1]
                compcor = configuration_df[1][2]
                n_compcor = configuration_df[1][3]
                if n_compcor == 'all':
                    pass
                elif n_compcor.isdigit():
                    configuration_df[1][3] = int(n_compcor)
                else:
                    raise ValueError

                condition1 = True if motion in ['basic', 'power2', 'derivatives', 'full'] else False
                condition2 = True if compcor in ['anat_combined', 'anat_separated', 'temporal', 'temporal_anat_combined', 'temporal_anat_separated'] else False

            elif denoise_strategy == preset_strategies[3]:
                # Conditions for wm_csf, global_signal
                wm_csf = configuration_df[1][1]
                global_signal = configuration_df[1][2]
                if global_signal == 'None':
                    global_signal = None
                    configuration_df[1][2] = global_signal
                    
                condition1 = True if wm_csf in ['basic', 'power2', 'derivatives', 'full'] else False
                condition2 = True if global_signal in ['basic', 'power2', 'derivatives', 'full', None] else False

            # Check conditions
            if condition1 and condition2:
                # Return the corrected df
                return configuration_df
            else:
                raise ValueError

        except ValueError:
                print(f"One or more values for the {denoise_strategy} denoise strategy parameters are not \
                available or incorrect. Please, check the configuration file and try again.")
                logging.error(f'One or more values for the {denoise_strategy} \
                                 denoise strategy parameters are not available or incorrect.')
                return None

    else:
        print(f'Denoising strategy "{denoise_strategy}" is not available.\
        Please, check the configuration file and choose one among {preset_strategies}')
        logging.error(f'The denoising strategy {denoise_strategy} is not available.')
        return None
    

def test_check_smoothing(configuration_df):
    
    '''
    Parameters:
        configuration_df: 'Pandas DataFrame Obj'
            dataframe containing the value for the smoothing FWHM parameter.
            
    Returns:
        configuration_df: 'Pandas DataFrame Obj'
            Same as the input dataframe, with numerical or 'None' strings replaced
            with int/float or None values.
    '''

    try:
        smoothing_fwhm = configuration_df[2][0]
        if smoothing_fwhm == 'None':
            configuration_df[2][0] = None
        elif smoothing_fwhm == 'fast':
            pass
        elif float(smoothing_fwhm) > 0:
            configuration_df[2][0] = float(smoothing_fwhm)
        else:
            raise ValueError
            
        return configuration_df
    
    except ValueError:
        print('Something is wrong with the smoothing_fwhm parameter.\
        Please, check the third column of the configuration file, and try again.')
        logging.error('Input value for smoothing_fwhm is incorrect.')
        return None
    
    
def test_check_filters(configuration_df):
    
    '''
    Parameters:
        configuration_df: 'Pandas DataFrame Obj'
            dataframe containing the values for the temporal filtering parameter.
            
    Returns:
        configuration_df: 'Pandas DataFrame Obj'
            Same as the input dataframe, with numerical or 'None' strings replaced
            with int/float or None values.
    ''' 
    
    try:
        high_pass = configuration_df[2][1]
        if high_pass == 'None':
            configuration_df[2][1] = None
        elif float(high_pass) > 0:
            configuration_df[2][1] = float(high_pass)
        else:
            raise ValueError

        low_pass = configuration_df[2][2]
        if low_pass == 'None':
            configuration_df[2][2] = None
        elif float(low_pass) > 0:
            configuration_df[2][2] = float(low_pass)
        else:
            raise ValueError
        
        return configuration_df
        
    except ValueError:
        print('Something is wrong with high_pass and low_pass parameters.\
        Please, check the third column of the configuration file, and try again.')
        logging.error('Input values for high_pass and low_pass are incorrect.')
        return None
