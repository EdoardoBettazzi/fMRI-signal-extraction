import argparse
import logging
from nilearn.interfaces.fmriprep import load_confounds_strategy
import numpy as np
import os
from pathlib import Path
from scipy import io

def user_args():

    # Arguments
    parser = argparse.ArgumentParser(description='Takes in input fMRIprep preprocessed data and extracts \
                                                    bold signals (mean values or first eigenvariate) of a chosen  \
                                                    number of subjects in a ROI-based or Seed-based manner. \
                                                    The output is given in a .txt (optionally .mat) file.')

    parser.add_argument("-i", "--input", help="Path to the fMRIprep derivatives directory, the dataset must be in BIDS-format \
                                                and contain a dataset_description.json file", required=True, metavar='')

    parser.add_argument("-config", "--configuration", help="csv file to specify a how to query the BIDS dataset, \
                                                            and how to perform the signal extraction. See README.MD for details.", required=True,metavar='' )

    parser.add_argument("-p", "--parcels", help="Specify a ROI parcellations atlas file.nii.gz", required=True, metavar='')
    
    parser.add_argument("-o", "--output", help="Path to the output directory where to save the extracted timeseries", required=True, metavar='')

    parser.add_argument("-mat", "--matlab", help="Optional. Save timeseries in .mat format", action="store_true", default=False)

    parser.add_argument("-csv", help="Optional. Save timeseries of all subjects in a single csv file", action="store_true", default=False)

    parser.add_argument("-DCM", help="Optional. Create a folder for DCM results and return a mat file containing the inputs \
                                     for running DCM analysis", action="store_true", default=False)

    args = parser.parse_args()

    return args


def get_scans(layout, BIDSquery_df):

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

    print('Parsing dataset.')

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


def load_custom_confounds(func_file, configuration_df):
 
    '''
    Parameters:
        func_file: 'str'
            path to subject's functional scan 
        configuration_df: 'Pandas DataFrame Obj'
            dataframe containing the verified information to perform the signal extraction.        
   
   Returns:
        confounds: 'Pandas DataFrame Obj'
            A reduced version of fMRIPrep confounds based on selected strategy and flags. 
            The columns contains the labels of the regressors. 
            (from nilearn.interfaces.fmriprep.load_confounds_strategy)
        sample_mask: None, 'numpy.ndarray', or 'list'
            When no volume requires removal, the value is None. 
            Otherwise, shape: (number of scans - number of volumes removed, )
            (from nilearn.interfaces.fmriprep.load_confounds_strategy, see nilearn docs for details)
    '''

    # Load preset denoise strategy
    strategy = configuration_df[3][0]
    strategies = ['simple', 'scrubbing', 'compcor', 'ica_aroma']

    if strategy == strategies[0]:
        confounds, sample_mask = load_confounds_strategy(func_file, 
                                                        denoise_strategy=strategy, 
                                                        motion=configuration_df[3][1],
                                                        wm_csf=configuration_df[3][2],
                                                        global_signal=configuration_df[3][3])

    elif strategy == strategies[1]:
        confounds, sample_mask = load_confounds_strategy(func_file, 
                                                        denoise_strategy=strategy, 
                                                        motion=configuration_df[3][1], 
                                                        scrub=configuration_df[3][2], 
                                                        fd_threshold=configuration_df[3][3], 
                                                        std_dvars_threshold=configuration_df[3][4],
                                                        global_signal=configuration_df[3][5])

    elif strategy == strategies[2]:
        confounds, sample_mask = load_confounds_strategy(func_file, 
                                                        denoise_strategy=strategy, 
                                                        motion=configuration_df[3][1],
                                                        compcor=configuration_df[3][2],
                                                        n_compcor=configuration_df[3][3])

    elif strategy == strategies[3]:
            confounds, sample_mask = load_confounds_strategy(func_file, 
                                                            denoise_strategy=strategy,
                                                            wm_csf=configuration_df[3][1], 
                                                            global_signal=configuration_df[3][2])
    

    return confounds, sample_mask
        
        
def output_file(output_folder, subj_signal, func_file, configuration_df, parcels, matlab=False):
    
    '''
    Parameters:
        output_folder: 'str'
            Path to the folder where to return the output file(s).
        subj_signal: 'numpy.ndarray'
            Subject's extracted bold signals.
        func_file: 'str'
            Path to subject's functional scan 
        configuration_df: 'Pandas DataFrame Obj'
            Dataframe containing all the necessary information to perform the signal extraction.
        parcels: 'str'
            Path to the parcellations atlas image. 
        matlab: 'bool'
            Set to True, to return the output in .mat format.
         
    Returns:
        txt (and optionally mat) file containing subject-specific signals.
    '''
    
    rois_labels = configuration_df[9].dropna().tolist()
    rois_names = configuration_df[11].dropna().tolist()
    
    if configuration_df[7][0] == 'SVD':
        reduction_strategy = 'first_eigenv'
    else:
        reduction_strategy = 'mean'
    
    # Save in .txt format
    txt_file = os.path.join(output_folder, f"{func_file.partition('func/')[2].partition('.nii')[0]}_{reduction_strategy}_signal.txt")
    atlas = parcels.split('/')[-1]
    with open(txt_file, 'w') as output_txt:
        output_txt.write(f'### HEADER ### ATLAS: {atlas}\n### HEADER ### Regions of interest (reported column-wise):\n{rois_labels}\n{rois_names}\n### HEADER ### Reduction strategy: {configuration_df[7][0]}\n')
        np.savetxt(output_txt, subj_signal)

    # Save in .mat file
    if matlab:
        file_mat = os.path.join(output_folder, f"{func_file.partition('func/')[2].partition('.nii')[0]}_{reduction_strategy}_signal.mat")
        header = f'{atlas}, {rois_names}, {configuration_df[7][0]}'
        io.savemat(file_mat, {'### HEADER ### ATLAS, Regions of interest, Reduction strategy': header,
                                'mean_timeseries': subj_signal})
        

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
    rois_labels = configuration_df[9].dropna().tolist()
    rois_names = configuration_df[11].dropna().tolist()

    # Save in mat format
    matfile = os.path.join(output_folder, 'dcm_inputs.mat')
    io.savemat(matfile,
            {'labels': rois_labels,
                'names': np.array(rois_names,dtype=object),
                'input_path': bold_signals_folder,
                'output_path': str(dcm_results_folder.absolute())},
                oned_as='column'
            )