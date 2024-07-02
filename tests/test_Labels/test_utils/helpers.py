import argparse
import logging
import numpy as np
import os
import pandas as pd
from pathlib import Path
from scipy import io 
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

## prompt for smoothing and filtering (low+high)

## prompt for reduction strategy

## prompt for roi_labels and roi_names
    
    

###### TESTED AND WORKING 

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

    parser.add_argument("-csv", help="Optional. Save timeseries of all subjects in a single csv file", action="store_true", default=False)
    
    parser.add_argument("-DCM", help="Optional. Return a mat file containing the inputs for running DCM analysis", action="store_true", default=False)

    args = parser.parse_args()

    return args


def test_get_scans(layout, BIDSquery_df):

    '''
    Parameters:
        layout: 'BIDSLayout Obj'
            Layout object representing an entire BIDS dataset.
        BIDSquery_df: 'Pandas DataFrame Obj'                
            Dataframe collecting the information necessary to retrieve the scans. 
            The first column MUST present values related to the following ['datatype' , 'task' ,'preprocessing', 'session', 'template']
            in this same exact order.

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


def dcm_inputs(configuration_df, output_folder, bold_signals_folder):

    '''
    Parameters:
        configuration_df: 'Pandas DataFrame Obj'
            Dataframe containing all the necessary information to perform the signal extraction.
        output_folder: 'str'
            Path to the folder where to return the output file(s).
        bold_signals_folder: 'str'
            Path to the folder where the extracted signals are stored.

    Returns:
        mat file containing input info for a DCM workflow.
    '''

    # Create output folder for DCM results
    dcm_results_folder = Path(os.path.join(output_folder, 'DCM_results'))
    dcm_results_folder.mkdir(parents=True, exist_ok=True)

    # Regions' labels and names
    rois = configuration_df[9].tolist()
    rois_names = configuration_df[11].tolist()

    # Save in mat format
    matfile = os.path.join(output_folder, 'dcm_inputs_test.mat')
    io.savemat(matfile,
            {'labels': rois,
                'names': np.array(rois_names,dtype=object),
                'input_path': bold_signals_folder,
                'output_path': str(dcm_results_folder.absolute())},
                oned_as='column'
            )