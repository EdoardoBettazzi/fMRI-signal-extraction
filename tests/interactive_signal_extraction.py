#!/usr/bin/env python3

# Imports
import argparse
from bids import BIDSLayout
import logging
from nilearn import datasets
from nilearn import image as nimg
from nilearn.interfaces.fmriprep import load_confounds_strategy
from nilearn.maskers import NiftiLabelsMasker
from nilearn import signal
import numpy as np
import os
import pandas as pd
from scipy import io
import scipy.linalg
from tqdm.auto import tqdm


# To query the BIDS dataset and get the functional scans paths
def get_func_files(derivatives_path, subjects_list=None):

    '''
    Parameters:
        derivatives_path            Path to the fMRIprep derivatives directory.
        subjects_list               csv file containing the list of subjects under analysis. None, by default.

    Returns:
        func_files                  List of file paths for the functional scans selected according to user prompts
    '''
    
    # Create Layout object to parse BIDS data struct
    layout = BIDSLayout(derivatives_path, absolute_paths=True, config=['bids','derivatives'])

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
    else:
        ses = None

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

    # Get functional files to load images
    func_files = layout.get(datatype='func',
                        task=task,
                        desc=desc,
                        session=ses,
                        space=space,
                        extension='nii.gz',
                        return_type='file')
    
    # Parse subjects list, if provided
    all_subjects = layout.get_subjects()
    func_files_set = []
    func_files_set_tolog = []
    if not subjects_list:
        # Include all subjects
        func_files_set = func_files
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
                            func_files_set.append(filepath)     
                            func_files_set_tolog.append(filepath.split('/sub-')[-1].partition('_')[0])       
                else:
                    logging.error(f'Subject {sub} is not in the list of subjects, or has no functional data.')

    print(f'{len(func_files_set)} subjects under analysis.')  

    # Log subjects  
    logging.info(f'Subjects under analysis:\n {func_files_set_tolog}\n')

    # Log the functional scans details
    log_df = pd.DataFrame(data = [[desc, task, ses, space]], columns=['description', 'task', 'session', 'template'])
    logging.info(f'functional scans details:\n {log_df}\n')

    return func_files_set

# To customize the parameters of the preset denoising strategies
def customize_confounds(strategy):

    '''
    Parameters:

        strategy                string indicating denoising strategy among {simple, scrubbing, compcor, ica_aroma}, 
                                    for the default options of each preset, check nilearn.interfaces.fmriprep.load_confounds_strategy
        
    Returns:
        config                  dictionary containing the value of the parameters for a given Nilearn preset denoising strategy
    '''

    # Available preset strategies
    strategies = ['simple', 'scrubbing', 'compcor', 'ica_aroma']

    # By default configuration, high_pass and demean are set True for all presets.                          # Demean value can be modified, are we interested in it???
    # All the parameters that remain N/A after customization are not available for the selected strategy. 
    config = {
        "Preset denoising strategy" : strategy,
        "high_pass" : "True",
        "motion" : "N/A",
        "wm_csf" : "N/A",
        "global_signal" : "N/A",
        "scrub" : "N/A",
        "fd_threshold" : "N/A",
        "std_dvars_threshold" : "N/A",
        "compcor" : "N/A",
        "n_compcor" : "N/A",
        "ica_aroma" : "N/A",
        "demean" : "True",
    }

    # Choose parameters settings according to the strategy
    if strategy == strategies[0]:

        while True:
            try:
                # Configure and check
                motion = str(input('Motion, choose among {basic, power2, derivatives, full}: '))
                wm_csf = str(input('WM and CSF, choose among {basic, power2, derivatives, full}: '))
                condition1 = True if all(value in ['basic', 'power2', 'derivatives', 'full'] for value in [motion, wm_csf]) else False
                if condition1 == False:
                    raise ValueError
                    
                # Configure and check
                global_signal = str(input('Global signal, choose among {basic, power2, derivatives, full, None}: '))
                if global_signal == 'None':
                    global_signal = None
                condition2 = True if global_signal in ['basic', 'power2', 'derivatives', 'full', None] else False
                if condition2 == False:
                    raise ValueError

                break
            except ValueError:
                print("\nNot valid.\n")
        
        # Save customization
        config["motion"] = motion
        config["wm_csf"] = wm_csf
        config["global_signal"] = global_signal

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

        config["motion"] = motion
        config["wm_csf"] = 'full'
        config["global_signal"] = global_signal
        config["scrub"] = scrub
        config["fd_threshold"] = fd_threshold
        config["std_dvars_threshold"] = std_dvars_threshold

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

        config["motion"] = motion
        config["compcor"] = compcor
        config["n_compcor"] = n_compcor

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

        config["wm_csf"] = wm_csf
        config["global_signal"] = global_signal
        config["ica_aroma"] = 'full'

    return config

# To load the confounds according to a preset strategy and 
# a given configuration of the parameters
def load_custom_confounds(func_nii_path, strategy, config):
 
    '''
    Parameters:
        func_nii_path           path to subject's functional image 
        strategy                string indicating denoising strategy among {simple, scrubbing, compcor, ica_aroma}, for the 
                                    default options of each preset, check nilearn.interfaces.fmriprep.load_confounds_strategy
        config                  dictionary containing the value of the parameters for a given Nilearn preset denoising strategy
        
    Returns:
        confound matrix                 pandas.DataFrame (timepoints)x(variable) with variable mean value in place of NaN
    '''

    # Customize strategy
    strategies = ['simple', 'scrubbing', 'compcor', 'ica_aroma']

    if strategy == strategies[0]:
        confounds, sample_mask = load_confounds_strategy(func_nii_path, 
                                                        denoise_strategy=strategy, 
                                                        motion=config['motion'],
                                                        wm_csf=config['wm_csf'],
                                                        global_signal=config['global_signal'])

    elif strategy == strategies[1]:
        confounds, sample_mask = load_confounds_strategy(func_nii_path, 
                                                        denoise_strategy=strategy, 
                                                        motion=config['motion'], 
                                                        scrub=config['scrub'], 
                                                        fd_threshold=config['fd_threshold'], 
                                                        std_dvars_threshold=config['std_dvars_threshold'],
                                                        global_signal=config['global_signal'])

    elif strategy == strategies[2]:
        confounds, sample_mask = load_confounds_strategy(func_nii_path, 
                                                        denoise_strategy=strategy, 
                                                        motion=config['motion'],
                                                        compcor=config['compcor'],
                                                        n_compcor=config['n_compcor'])

    elif strategy == strategies[3]:
            confounds, sample_mask = load_confounds_strategy(func_nii_path, 
                                                            denoise_strategy=strategy,
                                                            wm_csf=config['wm_csf'], 
                                                            global_signal=config['global_signal'])
    
    return confounds, sample_mask
        
# To extract the first eigenvariate
def extract_eig(A):
    
    '''
    Parameters:

        A                       matrix of shape (m)x(n), where m is the number of timesteps and n number of voxels.
        
    Returns:
        first_eig               First normalized eigenvariate
    '''

    # Shape of A
    m = A.shape[0]
    n = A.shape[1]
    
    At = np.transpose(A)
    
    if m > n:
        # do i need to make A sqrt like A.t * A ??? 
        # Perform Singular Value Decomposition
        U, S, V = scipy.linalg.svd(At @ A,
                                full_matrices=False,
                                compute_uv=True,
                                check_finite=False,
                                overwrite_a=True,
                                #lapack_driver='gesvd' #driver used by matlab
                            )

        # Extract first (largest) right eigenvector
        v = V[:,0]
        # Weighted sum, normalized to first sing val
        u = A @ v/S[0]**.5
    
    else: 
        U, S, V = scipy.linalg.svd(A @ At,
                            full_matrices=False,
                            compute_uv=True,
                            check_finite=False,
                            overwrite_a=True,
                            #lapack_driver='gesvd'
                        )

        # Extract first (largest) left eigenvector
        u = U[:,0]
        # Weighted sum, normalized to first sing val
        v = At @ u/S[0]**.5
        
    # Compute sign of the sum of v (WHY???)     
    d = np.sign(np.sum(v))
    # And multiply
    v = v * d
    u = u * d
    
    # Compute normalized first eigenvariate
    Y = u * (S[0]/n)**.5
    
    # Normalize
    Y = (Y - np.mean(Y)) / np.std(Y)
    
    # Return first eigenvariate 
    return Y


if __name__ == "__main__":

    # Arguments
    parser = argparse.ArgumentParser(description='Takes in input fMRIprep minimally preprocessed data and extracts \
                                                    the timeseries (mean values or first eigenvariate) of a chosen  \
                                                    number of subjects in a .txt (optionally .mat) file.')

    parser.add_argument("-i", "--input", help="Path to the fMRIprep derivatives directory, the dataset must be in BIDS-format \
                                                and contain a dataset_description.json file", required=True, metavar='')
    
    parser.add_argument("-s", "--subjects", help="Optional. csv file with the numbers of the subjects from which you want to extract timeseries. \
                                            if not specified, timeseries are extracted for all the subjects.")

    parser.add_argument("-p", "--parcels", help="Optional. Specify a ROI parcellation atlas file.nii.gz, \
                                                or by default the Schaeffer2018 7Networks atlas is used", metavar='')
    
    parser.add_argument("-r", "--rois", help="Optional. csv file with the labels (first column) of the ROIs, from which you want to extract timeseries. \
                                            if not specified, timeseries are extracted from all the ROIs in the atlas")
    
    parser.add_argument("-o", "--output", help="Path to the output directory where to save the extracted timeseries", required=True, metavar='')

    parser.add_argument("-mat", "--matlab", help="Optional. Save timeseries in .mat format", action="store_true", default=False)

    args = parser.parse_args()

    # Set up logging
    logging.basicConfig(filename=os.path.join(args.output,'config.log'),
                        filemode='w',
                        level=logging.INFO,
                        format='%(asctime)s:%(levelname)s:%(message)s')   

    ## LOAD DATASET, ATLAS AND LIST OF ROIs

    # Path to fMRIprep derivative folder, containing minimally preprocessed Nifti.gz data and tsv confounds table
    derivatives_path = args.input

    # Parse BIDS dataset to get functional scans
    print('Parse BIDS dataset to get the functional data.')
    subjects_list = args.subjects
    func_files = get_func_files(derivatives_path, subjects_list)

    # Load parcellations
    # Manually
    if args.parcels:
        print('Loading parcellations.')
        parcellations = args.parcels
        atlas = nimg.load_img(parcellations)       
    else:
        # Or load Schaefer18 parcellations from Nilearn
        print('\nFetch Schaefer 2018 parcellation atlas.')
        n_rois = int(input('Number of ROI in the atlas? '))
        resolution_mm = int(input('Spatial resolution of atlas image (1, 2)? '))
        fetched = datasets.fetch_atlas_schaefer_2018(n_rois=n_rois, resolution_mm=resolution_mm)
        # Prepend background label
        fetched.labels = np.insert(fetched.labels, 0, 'Background')
        # Load Nifti image
        parcellations = fetched.maps
        atlas = nimg.load_img(parcellations)


    # Resample atlas to the functional image
    f_img = nimg.load_img(func_files[0])
    resampled_atlas = nimg.resample_img(atlas,
                            target_affine=f_img.affine,
                            target_shape=f_img.shape[:3],
                            interpolation='nearest')

    # get atlas data
    atlas_data = resampled_atlas.get_fdata()

    # Extract ROIs labels
    labels = np.unique(atlas.get_fdata())
    print('')

    logging.info(f'PARCELLATIONS:\n {parcellations}\n')

    # Parse ROIs list, if provided
    if not args.rois:
        # Include all labels, except background
        rois = labels[1:]
        logging.info('Regions of interest under analysis: ALL \n')
    else:    
        with open(args.rois, 'r') as f:
            roi_df = pd.read_csv(f, header=None)
            # Sanity check
            if pd.api.types.is_numeric_dtype(roi_df.iloc[:,0]):
                rois = sorted(roi_df.iloc[:,0])
            else:
                raise ValueError('One or more ROI labels are not integers.')
            
            if  not (rois.min() >= 0 and rois.max() <= len(labels[1:])):
                raise ValueError('One or more ROI labels are not in the selected atlas.')
        logging.info(f'Regions of interests under analysis:\n {list(rois)}\n')

    ## CUSTOMIZE CONFOUNDS REMOVAL

    # Choose denoising strategy
    desc = func_files[0].partition('desc-')[2].partition('_')[0]
    if desc == 'smoothAROMAnonaggr':
        denoise_strategy = 'ica_aroma'
    else:
        while True:
            try:
                denoise_strategy = str(input('Select preset denoising strategy among (simple, scrubbing, compcor, ica_aroma): '))
                if denoise_strategy not in ['simple', 'scrubbing', 'compcor', 'ica_aroma']:
                    raise ValueError
                break
            except ValueError:
                print("\nNot available.\n")

    # Customize
    config = customize_confounds(denoise_strategy)

    # Define smoothing TODO
    while True:
        try:
            smooth = input('Smoothing kernel (in mm, or None): ')
            if smooth == 'None':
                smooth = None
            elif smooth.isdigit():
                smooth = int(smooth)
            else:
                raise ValueError
            break

        except ValueError:
            print("\nPlease, type an integer value.\n")

    config['smoothing'] = smooth

    # Define temporal filtering TODO
    while True:
            try:
                high_pass=input('High cutoff frequenzy (in Hz, or None): ')
                if high_pass == 'None':
                    high_pass = None
                elif high_pass.isdigit():
                    high_pass = float(high_pass)
                else:
                    raise ValueError
                break
            except ValueError:
                print("\nPlease, type an integer value.\n")

    while True:
            try:
                low_pass=input('Low cutoff frequenzy (in Hz, or None): ')
                if low_pass == 'None':
                    low_pass = None
                elif low_pass.isdigit():
                    low_pass = float(low_pass)
                else:
                    raise ValueError
                break
            except ValueError:
                print("\nPlease, type an integer value.\n")

    config['high pass filtering'] = high_pass
    config['low pass filtering'] = low_pass

    # Summing up
    config = {key:val for key, val in config.items() if val != 'N/A'}
    config_df = pd.DataFrame.from_dict(config, orient='index', columns=['Configuration'])
    logging.info(f'Confounds removal config:\n {config_df}\n')
    print('')
    print(config_df)
    print('')

    ## TIMESERIES EXTRACTION

    # Choose signal reduction strategy
    red_strategies = [
        'mean',
        'median',
        'sum',
        'minimum',
        'maximum',
        'standard_deviation',
        'variance', 
        'SVD'
    ]
    
    while True:
        try: 
            reduction_strategy = str(input(f'Select reduction strategy to perform during extraction among {red_strategies}: '))
            if reduction_strategy not in red_strategies:
                raise ValueError
            break
        except ValueError:
            print("\nNot available.\n")

    logging.info(f'Signal reduction strategy: {reduction_strategy}\n')

    if reduction_strategy in red_strategies[:-1]:
            
        # AS REDUCED SIGNALS

        # Create Masker Object for signal extraction
        masker = NiftiLabelsMasker(labels_img=atlas,
                                    standardize=True,
                                    smoothing_fwhm=smooth, 
                                    low_pass=low_pass,
                                    high_pass=high_pass,
                                    # t_r=layout.get_tr(func_files[0]), not available for some files, can be omitted?
                                    # memory='nilearn_cache',     cache removed to test for performance
                                    verbose=0,
                                    strategy=reduction_strategy)

        # Apply
        print('')
        num_subj = len(func_files)
        for i in tqdm(range(0, num_subj), desc=f'Extracting {reduction_strategy} signals', position=0, leave=True): 

            # Load confounds in a customized way
            confounds, sample_mask = load_custom_confounds(func_files[i], denoise_strategy, config)

            # Regress out and extract
            subj_signal = masker.fit_transform(func_files[i], confounds=confounds, sample_mask=sample_mask)

            # Save in txt file
            output_txt = os.path.join(args.output, f"{func_files[i].partition('func/')[2].partition('.nii')[0]}_mean_timeseries.txt")
            # with open(output_txt, 'w') as output:
            #     atlas_header = parcellations.split('/')[-1]
            #     output.write(f'### HEADER ### ATLAS: {atlas_header}; Regions of interest: {list(rois)}' )
            np.savetxt(output_txt, subj_signal)

            # Save in .mat file
            if args.matlab:
                io.savemat(os.path.join(args.output, f"{func_files[i].partition('func/')[2].partition('.nii')[0]}_mean_timeseries.mat"), {'mean_timeseries': subj_signal})

    elif reduction_strategy == red_strategies[-1]:

        # OR AS FIRST EIGENVARIATE

        num_subj = len(func_files)
        for i in tqdm(range(0, num_subj), desc='Extracting first eigenvariate', position=0, leave=True): 

            # Subject data
            subj = func_files[i].partition('sub-90')[2].partition('/')[0]
            f_img = nimg.load_img(func_files[i])
            f_data = f_img.get_fdata()

            # Load confounds in a customized way
            confounds, sample_mask = load_custom_confounds(func_files[i], denoise_strategy, config)

            # Create subject array w/ shape (timepoints)x(n of ROIs)
            timepoints = f_data.shape[3]
            if sample_mask is not None:
                timepoints = sample_mask.shape[0]
            subj_first_eig_signal = np.zeros(shape=(timepoints, len(rois)))

            for j in range(len(rois)):
        
                # Make boolean array based on roi
                roi_condition = np.where(atlas_data==rois[j], True, False)

                # Boolean indexing
                roi_timeseries = f_data[roi_condition]

                # Swap axes
                roi_timeseries = np.rollaxis(roi_timeseries, -1)

                # Regress out confounds and clean                                   ------> ATTENTION: NOT PERFORMING SMOOTHING OR TEMPORAL FILTERING
                cleaned_roi_timeseries = signal.clean(roi_timeseries, 
                                                        detrend=False,
                                                        standardize=True,
                                                        sample_mask=sample_mask,
                                                        confounds=confounds)

                # SVD and extract first eigenvariates
                roi_eig = extract_eig(cleaned_roi_timeseries)

                # Add to subject array 
                subj_first_eig_signal[:, j] = roi_eig

            # Save in txt file
            np.savetxt(os.path.join(args.output, f"{func_files[i].partition('func/')[2].partition('.nii')[0]}_firsteig_timeseries.txt"), subj_first_eig_signal)

            # Save in .mat file
            if args.matlab:
                io.savemat(os.path.join(args.output, f"{func_files[i].partition('func/')[2].partition('.nii')[0]}_firsteig_timeseries.mat"), {'firsteig_timeseries': subj_first_eig_signal})


    '''
    ####### Performance evaluation
    from timeit import default_timer as timer

    perf_test = [1, 5, 10, 20]
    perf_test= [40]

    for test in perf_test:

        num_subs = test

        # Get resting state data (preprocessed, mask, and confounds file) for the selected number of subjects
        func_files = layout.get(datatype='func', 
                                task='rest',
                                desc='preproc',
                                space='MNI152NLin2009cAsym',
                                extension='nii.gz',
                                return_type='file')[:num_subs]

        from timeit import default_timer as timer  
        start = timer()

        print()
        print(stop - start)
        print()

        with open(args.output+'perf.txt', "a+") as perf_file:

            perf_file.write(f'Time required for strategy {denoise_strategy} on {num_subs} subjects is {stop - start}')
            perf_file.write('\n')
    '''
