import logging
from nilearn.interfaces.fmriprep import load_confounds_strategy
import nilearn.image as nimg
from nilearn.maskers import NiftiLabelsMasker
from nilearn import signal
import numpy as np
import os
import pandas as pd
from pathlib import Path
from scipy import linalg
from scipy import io
from tqdm.auto import tqdm

######### TO BE TESTED 

def test_rois_mean(func_files, configuration_df, atlas, output_folder, rois_list, matlab=False):
    
    '''
    Parameters:
        func_files: 'list'
            List of paths to the functional scans.
        configuration_df: 'Pandas DataFrame Obj'
            Dataframe containing all the necessary information to perform the signal extraction.
        atlas_img: 'Nifti image'
            Parcellation atlas image.
        rois_list: 'csv'
            csv file containing the list of Regions of interest, from which you want to extract
            the bold signals.
         
    Returns:
        'txt' and/or 'mat' files, as many as the number of scans in the input list, each containing
        subject-specific signals extracted as mean.   
    '''

    # Get labels for the provided Atlas 
    labels = np.unique(atlas.get_fdata())
    
    # Read the list of ROIs
    try:
        with open(rois_list, 'r') as f:      
            roi_df = pd.read_csv(f, header=None)
            
            # Sanity check
            if pd.api.types.is_numeric_dtype(roi_df.iloc[:,0]):
                rois = list(sorted(roi_df.iloc[:,0]))
            else:
                raise ValueError('One or more ROI labels are not integers.')
            
            if  not (rois.min() >= 0 and rois.max() <= len(labels[1:])):
                raise ValueError('One or more ROI labels are not in the selected atlas.')
                
        logging.info(f'Regions of interests under analysis:\n {rois}\n')
        
    except FileNotFoundError:
        print(f"Something's wrong with the ROIs csv file, please check that {rois_list} is the correct file.")
        logging.error(f'Failed in retrieving or reading the input file: {rois_list}')

    # Load maskers' parameters
    smoothing_fwhm = None
    low_pass = None
    high_pass = None
    preprocessing = func_files[0].partition('desc-')[2].partition('_')[0]
    if preprocessing == 'preproc':
        # Load maskers' parameters
        smoothing_fwhm = configuration_df[5][0]
        low_pass = configuration_df[5][1]
        high_pass = configuration_df[5][2]
    
    # Extract
    num_subj = len(func_files)
    for i in tqdm(range(0, num_subj), desc='Extracting first eigenvariate', position=0, leave=True): 

        # Subject data
        f_img = nimg.load_img(func_files[i])
        f_data = f_img.get_fdata()

        # Resample the atlas to the functional scan
        resampled_atlas = nimg.resample_img(atlas,
                            target_affine=f_img.affine,
                            target_shape=f_img.shape[:3],
                            interpolation='nearest')

        # Load confounds in a customized way
        confounds, sample_mask = test_load_custom_confounds(func_files[i], configuration_df)

        # Create subject array w/ shape (timepoints)x(n of ROIs)
        timepoints = f_data.shape[3]
        if sample_mask is not None:
            timepoints = sample_mask.shape[0]
        subj_mean_signal = np.zeros(shape=(timepoints, len(rois)))

        for r in range(len(rois)):

            # Mask image with math_img
            roi_img = nimg.math_img(f"img == {rois[r]}", img=resampled_atlas)

            # Create Masker Object for signal extraction
            masker = NiftiLabelsMasker(labels_img=roi_img,
                                        standardize=True,
                                        smoothing_fwhm=smoothing_fwhm, 
                                        low_pass=low_pass,
                                        high_pass=high_pass,
                                        # t_r=layout.get_tr(func_files[0]), not available for some files, can be omitted?
                                        memory='nilearn_cache',
                                        verbose=0
                                        )
            
            # Perform extraction
            masker.fit()
            roi_mean = masker.transform(f_img,
                                    sample_mask=sample_mask,
                                    confounds=confounds)

            # Add to subject array 
            subj_mean_signal[:, r] = roi_mean
            
            # Output file
            test_output_file(output_folder, subj_mean_signal, func_files[i], configuration_df, atlas, matlab=matlab)





######### TESTED

def test_load_custom_confounds(func_file, configuration_df):
 
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


def test_mean(func_files, configuration_df, atlas, output_folder, matlab=False):
    
    '''
    Parameters:
        func_files: 'list'
            List of paths to the functional scans.
        configuration_df: 'Pandas DataFrame Obj'
            Dataframe containing all the necessary information to perform the signal extraction.     
        atlas_img: 'str'
            Path to the parcellations atlas.
        output_folder: 'str'
            Path to the folder where to return the output file(s).
        matlab: 'bool'
            Set to True, to return the output in .mat format too.
         
    Returns:
        'txt' (and optionally 'mat') files, as many as the number of scans in the input list, each containing
        subject-specific signals extracted as mean.   
    '''

    # Get Atlas image and labels
    atlas_img = nimg.load_img(atlas)
    atlas_data = atlas_img.get_fdata()
    labels = np.unique(atlas_data)

    # Regions under analysis
    rois_series = configuration_df[9][:]
    rois_names = configuration_df[11][:]

    # Sanity checks
    if len(rois_series) != len(rois_names):
        raise ValueError('ROIs labels and names do not match.')
    
    if rois_series.isnull().values.any() or rois_names.isnull().values.any():
        rois_series = rois_series.dropna()
        rois_names = rois_names.dropna()

    if pd.api.types.is_numeric_dtype(rois_series):
        rois = rois_series.tolist()
    else:
        raise ValueError('One or more ROI labels are not integers.')
    
    for r in rois:
        if  r not in labels:
            raise ValueError('One or more ROI labels are not in the selected atlas.')
    
    logging.info(f'Regions of interest under analysis:\n {rois}\n')

    # If the ROIs are a subset of the Atlas ROIs, create a new Nifti image with the subset
    if len(rois) < len(labels):

        # Initialize Mask of False values with shape == atlas.shape\
        subset_mask = np.zeros((atlas_data.shape), dtype=bool)

        # Produce a single boolean mask based on multiple parcels
        for i in rois:
            msk = np.where(atlas_data == i, i, 0)
            subset_mask = subset_mask + msk
            
        # Make a new image based on the Atlas and the Regions of Interest
        subset_img = nimg.new_img_like(atlas, subset_mask)
        atlas_img = subset_img

    
    # Load maskers' parameters
    smoothing_fwhm = None
    low_pass = None
    high_pass = None
    preprocessing = func_files[0].partition('desc-')[2].partition('_')[0]
    if preprocessing == 'preproc':
        # Load maskers' parameters
        smoothing_fwhm = configuration_df[5][0]
        low_pass = configuration_df[5][1]
        high_pass = configuration_df[5][2]

    
    # Create Masker Object for signal extraction
    masker = NiftiLabelsMasker(labels_img=atlas_img,
                                standardize=True,
                                smoothing_fwhm=smoothing_fwhm, 
                                low_pass=low_pass,
                                high_pass=high_pass,
                                # t_r=layout.get_tr(func_files[0]),
                                memory='nilearn_cache',
                                verbose=0
                              )

    # Apply
    print('')
    num_subj = len(func_files)
    for i in tqdm(range(0, num_subj), desc=f'Extracting mean signals', position=0, leave=True): 

        # Load confounds in a customized way
        confounds, sample_mask = test_load_custom_confounds(func_files[i], configuration_df)

        # Regress out and extract
        subj_mean_signal = masker.fit_transform(func_files[i], confounds=confounds, sample_mask=sample_mask)

        # Output file
        test_output_file(output_folder, subj_mean_signal, func_files[i], configuration_df, atlas, matlab=matlab)


def test_extract_eig(A):
    
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
        U, S, V = linalg.svd(At @ A,
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
        U, S, V = linalg.svd(A @ At,
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


def test_first_eig(func_files, configuration_df, atlas, output_folder, matlab=False, csv=True):
    
    """
    Parameters:
        func_files: 'list'
            List of paths to the functional scans.
        configuration_df: 'Pandas DataFrame Obj'
            Dataframe containing all the necessary information to perform the signal extraction.
        atlas: 'str'
            Path to the parcellation atlas image.
        output_folder: 'str'
            Path to the folder where to return the output file(s).
        matlab: 'bool'
            Set to True, to return the output in .mat format too.
        csv: 'bool'
            Set to True, to return a csv file containing signals from all subjects.
         
    Returns:
        'txt' and/or 'mat' files, as many as the number of scans in the input list, each containing
        subject-specific signals extracted as first eigenvariate.            
    """
    
    # Get Atlas image and labels
    atlas_img = nimg.load_img(atlas)
    atlas_data = atlas_img.get_fdata()
    labels = np.unique(atlas_data)

    # Regions under analysis
    rois_series = configuration_df[9][:]
    rois_names = configuration_df[11][:]

    # Sanity checks
    if len(rois_series) != len(rois_names):
        raise ValueError('ROIs labels and names do not match.')
    
    if rois_series.isnull().values.any() or rois_names.isnull().values.any():
        rois_series = rois_series.dropna()
        rois_names = rois_names.dropna()
        
    if pd.api.types.is_numeric_dtype(rois_series):
        rois = rois_series.tolist()
    else:
        raise ValueError('One or more ROI labels are not integers.')
    
    for r in rois:
        if  r not in labels:
            raise ValueError('One or more ROI labels are not in the selected atlas.')
    
    logging.info(f'Regions of interest under analysis:\n {rois}\n')

    # Load masker parameters
    smoothing_fwhm = None
    low_pass = None
    high_pass = None
    preprocessing = func_files[0].partition('desc-')[2].partition('_')[0]
    if preprocessing == 'preproc':
        # Load maskers' parameters
        smoothing_fwhm = configuration_df[5][0]
        low_pass = configuration_df[5][1]
        high_pass = configuration_df[5][2]
    
    ## Extract subjects' signals
    num_subj = len(func_files)

    # Initialize dataframe to output all subjects signals in csv
    all_subjects_df = pd.DataFrame(columns=[['Subject']+rois])
    
    for i in tqdm(range(0, num_subj), desc='Extracting first eigenvariate', position=0, leave=True): 

        # Load subject image
        f_img = nimg.load_img(func_files[i])

        # Smooth
        if smoothing_fwhm is None:
            pass
        else:
            f_img = nimg.smooth_img(f_img, smoothing_fwhm)
            
        f_data = f_img.get_fdata()
        
        # Resample the atlas to the functional scan
        resampled_atlas = nimg.resample_img(atlas_img,
                            target_affine=f_img.affine,
                            target_shape=f_img.shape[:3],
                            interpolation='nearest')
        resampled_atlas_data = resampled_atlas.get_fdata()

        # Load confounds in a customized way 
        confounds, sample_mask = test_load_custom_confounds(func_files[i], configuration_df)

        # initialize subject array w/ shape (timepoints)x(n of ROIs)
        timepoints = f_data.shape[3]
        if sample_mask is not None:
            timepoints = sample_mask.shape[0]
        subj_first_eig_signal = np.zeros(shape=(timepoints, len(rois)))

        for r in range(len(rois)):

            # Make boolean array based on roi
            roi_condition = np.where(resampled_atlas_data==rois[r], True, False)

            # Boolean indexing
            roi_timeseries = f_data[roi_condition]

            # Swap axes
            roi_timeseries = np.rollaxis(roi_timeseries, -1)

            # Regress out confounds and clean                      
            cleaned_roi_signals = signal.clean(roi_timeseries, 
                                                    detrend=False,
                                                    standardize=True,
                                                    sample_mask=sample_mask,
                                                    high_pass=high_pass,
                                                    low_pass=low_pass,
                                                    confounds=confounds)

            # SVD and extract first eigenvariates
            roi_eig = test_extract_eig(cleaned_roi_signals)

            # Add to subject array 
            subj_first_eig_signal[:, r] = roi_eig

            # Output file
            test_output_file(output_folder, subj_first_eig_signal, func_files[i], configuration_df, atlas, matlab=matlab)

            # Subject dataframe
            if csv:
                subj_df = pd.DataFrame(data=subj_first_eig_signal, columns=[rois])
                subj = func_files[i].partition('func/')[2].partition('_')[0]
                subj_df.insert(0, 'Subject', subj)

        # Concatenate subjects dataframes
        if csv:
            all_subjects_df = pd.concat([all_subjects_df, subj_df], axis=0)
    
    # Save csv file
    if csv:
        output_csv = os.path.join(output_folder, 'all_subjects_signals.csv')
        all_subjects_df.to_csv(output_csv , sep=',', index=False)


def test_output_file(output_folder, subj_signal, func_file, configuration_df, parcels, matlab=False):
    
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
        mat file containing the
    '''
    
    rois_labels = configuration_df[9].tolist()
    rois_names = configuration_df[11].tolist()
    
    if configuration_df[7][0] == 'SVD':
        reduction_strategy = 'first_eigenv'
    else:
        reduction_strategy = 'mean'

    # Save in .txt format
    txt_file = os.path.join(output_folder, f"{func_file.partition('func/')[2].partition('.nii')[0]}_{reduction_strategy}_signal.txt")
    atlas = parcels.split('/')[-1]
    with open(txt_file, 'w+') as output_txt:
        output_txt.write(f'### HEADER ### ATLAS: {atlas}\n### HEADER ### Regions of interest (reported column-wise):\n{rois_labels}\n{rois_names}\n### HEADER ### Reduction strategy: {configuration_df[7][0]}\n')
        np.savetxt(output_txt, subj_signal)

    # Save in .mat file
    if matlab:
        file_mat = os.path.join(output_folder, f"{func_file.partition('func/')[2].partition('.nii')[0]}_{reduction_strategy}_signal.mat")
        header = f'{atlas}, {rois_names}, {configuration_df[7][0]}'
        io.savemat(file_mat, {'### HEADER ### ATLAS, Regions of interest, Reduction strategy': header,
                                'mean_timeseries': subj_signal})

