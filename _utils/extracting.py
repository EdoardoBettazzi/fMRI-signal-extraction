from _utils import helpers
import logging
import nilearn.image as nimg
from nilearn.maskers import NiftiLabelsMasker
from nilearn import signal
import numpy as np
import os 
import pandas as pd
from scipy import linalg, io 
from tqdm.auto import tqdm


def mean(func_files, configuration_df, atlas, output_folder, matlab=False, csv=False):
    
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
        csv: 'bool'
            Set to True, to return a csv file containing signals from all subjects.

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
            raise ValueError(f'ROI label {r} is not in the selected atlas.')
    
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

        # Use the new image as atlas
        atlas_img = subset_img

    # Initialize dataframe to output all subjects signals in csv
    if csv:
        all_subjects_df = pd.DataFrame(columns=[['Subject']+rois])
    
    # Smooth or filter, only if data is minimally processed
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
                                high_pass=high_pass,
                                low_pass=low_pass,
                                memory='nilearn_cache',
                                verbose=0
                              )

    # Apply
    print('')
    num_subj = len(func_files)
    for i in tqdm(range(0, num_subj), desc=f'Extracting mean signals', position=0, leave=True): 

        # Load confounds in a customized way
        confounds, sample_mask = helpers.load_custom_confounds(func_files[i], configuration_df)

        # Regress out and extract
        subj_mean_signal = masker.fit_transform(func_files[i], confounds=confounds, sample_mask=sample_mask)

        # Output file
        helpers.output_file(output_folder, subj_mean_signal, func_files[i], configuration_df, atlas, matlab=matlab)

        # Subject dataframe
        if csv:
            subj = func_files[i].partition('func/')[2].partition('_')[0]
            subj_df = pd.DataFrame(data=subj_mean_signal, columns=[rois])
            subj_df.insert(0, 'Subject', subj)

        # Concatenate subjects dataframes
        if csv:
            all_subjects_df = pd.concat([all_subjects_df, subj_df], axis=0)

    # Save csv file
    if csv:
        output_csv = os.path.join(output_folder, 'all_subjects_signals.csv')
        all_subjects_df.to_csv(output_csv , sep=',', index=False)


def first_eig(func_files, configuration_df, atlas, output_folder, matlab=False, csv=False):
    
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
            raise ValueError(f'ROI label {r} is not in the selected atlas.')

    logging.info(f'Regions of interest under analysis:\n {rois}\n')

    # Smooth or filter, only if data is minimally processed
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

    # Initialize dataframe to output all subjects signals in csv
    if csv:
        all_subjects_df = pd.DataFrame(columns=[['Subject']+rois])

    num_subj = len(func_files)
    for i in tqdm(range(0, num_subj), desc='Extracting first eigenvariate', leave=True): 

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
        confounds, sample_mask = helpers.load_custom_confounds(func_files[i], configuration_df)

        # Create subject array w/ shape (timepoints)x(n of ROIs)
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
                                                    high_pass=high_pass,
                                                    low_pass=low_pass,
                                                    sample_mask=sample_mask,
                                                    confounds=confounds)

            # SVD and extract first eigenvariates
            roi_eig = extract_eig(cleaned_roi_signals)

            # Add to subject array 
            subj_first_eig_signal[:, r] = roi_eig

            # Output file
            helpers.output_file(output_folder, subj_first_eig_signal, func_files[i], configuration_df, atlas, matlab=matlab)

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
        
    # Compute sign of the sum of v 
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
