# Signal Extraction with Nilearn
The following tool can be used to perform a customized extraction of BOLD signals from BIDS-formatted datasets of fMRI scans preprocessed with fMRIPrep (either minimally or with ICA_AROMA). See below for details on the customization.

### Setup

In order to read a BIDS-formatted dataset, this tool would require a [dataset_description.json](https://github.com/EdoardoBettazzi/fMRI-signal-extraction/blob/main/dataset_description.json) file. In case the file is not present, an empty dataset_description.json is created. 
Still, you're encouraged to add a complete one for the sake of reproducible science (check [BIDS recommendations](https://bids-specification.readthedocs.io/en/stable/modality-agnostic-files.html#dataset_descriptionjson) for more details).

## Installation

It is strongly recommended to use a virtual environment and install the packages inside it. 

If you use the package manager [conda](https://docs.conda.io/en/latest/), creating a virtual environment for a set of packages is as easy as

```bash
conda env create -f environment.yml
```
The command above will create an environment named as specified in the ```yml``` file, in our case it will be called ```signal_extraction_env```. The next step is to activate it, by executing

```bash
conda activate signal_extraction_env
```
For a user-guide and more details on conda environments, check [Managing environments](https://conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html#managing-environments). 

If you use the package manager [pip](https://pypi.org/project/pip/),  check [here](https://packaging.python.org/en/latest/tutorials/installing-packages/#creating-virtual-environments) if you intend to create a virtual environment, otherwise the packages can be installed by executing:

```bash
python3 -m pip install -r requirements.txt
```
## Customize the extraction
To customize the extraction, create a `configuration.csv`
 file as follows:

|                   |           |                           |          |                     |                 |                      |             | 
|:--------------    |:-------:  |:-----------------         | :------: | -------------:      | :-----:         | :-------------       |:----------: |
|**DATATYPE**       | e.g func  |**PRESET DENOISE STRATEGY**| See below |**SMOOTHING FWHM**  | Number, or None |**REDUCTION STRATEGY**| mean or SVD |
|**TASK**           | e.g rest  |**CONFOUNDS 1**            | ...       |**HIGH PASS FILTER**| Number, or None |                      |             |
|**PREPROCESSING**  | ...       |**CONFOUNDS 2**            | ...       |**LOW PASS FILTER** | Number, or None |                      |             |
|**SESSION**        | ...       |**CONFOUNDS 3**            | ...       |                    |                 |                      |             |
|**BRAIN TEMPLATE** | ...       |**CONFOUNDS 4**            | ...       |                    |                 |                      |             |

|                               |                       |                      |                                         |
| :-------------                |:-------------:        |  :-----              | :----------:                            |
|**ROI – LABELS/COORDINATES**   | e.g 24 or -58,-38,30  |**ROI – ANAT.NAMES**  | e.g 7Networks_LH_SalVentAttn_ParOper_1  | 
|                               | ...                   |                      | ...                                     | 
|                               | ...                   |                      | ...                                     |                
|                               | ...                   |                      | ...                                     |   
|                               | ...                   |                      | ...                                     |              

**1st and 2nd columns** specify the parameters for querying the ```BIDS dataset``` given in input and retrieve the fMRI scans.

**3rd and 4th columns** specify preset strategy and parameters for the confounds regression. See [nilearn.interfaces.fmriprep.load_confounds_strategy](https://nilearn.github.io/stable/modules/generated/nilearn.interfaces.fmriprep.load_confounds_strategy.html) for the available preset strategies.
See [nilearn.interfaces.fmriprep.load_confounds](https://nilearn.github.io/stable/modules/generated/nilearn.interfaces.fmriprep.load_confounds.html) for details on the confounds.

**5th and 6th columns** specify values for smoothing the image, and for the higher and lower thresholds for temporal filtering.   
For details on the smoothing parameter see ```fwhm``` in [nilearn.image.smooth_img](https://nilearn.github.io/stable/modules/generated/nilearn.image.smooth_img.html#).  
For details on butterworth filtering, check ```high_pass``` and ```high_pass``` in [nilearn.signal.clean](https://nilearn.github.io/dev/modules/generated/nilearn.signal.clean.html)

**7th and 8th columns** allow to select a strategy to summarize regional signal between ```mean``` and ```SVD```.

**9th and 10th columns** are used by the program to select, in the given parcellation atlas, the Regions of interest from which it will extract the signals.

**11th and 12th columns** are used in the header of the output file for referencing clearly the ROI labels/coordinates.

In each cell, replace a value for the reported parameter. Please notice that each parameter is necessary, therefore all the specified cells must have a value (specify ```None```, if you want a parameter not to be used).

## Usage

```bash
python3 nilearn_signal_extraction.py \
--input path/to/your/fMRIPrep/dataset \
--output path/to/output/folder \
--configuration Configuration.csv \
--parcels  ParcellationAtlas.nii.gz \
[--matlab]
[--DCM ]
```

In the output folder, the script will replicate the BIDS-formatted derivatives directory tree for each subject. Where the derivatives were stored in the original dataset,a txt (and .mat, if specified in the command) file is created, containing the signals extracted for the specified ROIs from the provided atlas.
Moreover, if the optional parameter `--DCM` is present, a directory named `DCM_results` and a mat file containing a struct with network details and input/output paths for applying Dynamic Causal Modeling (DCM), are created in the same output directory.


The output txt file presents a header in the first 3 lines, reporting some basic informations on the file contents (see below).  
  
```### HEADER ### ATLAS:               ```  
```### HEADER ### Regions of interest: ```  
```### HEADER ### Reduction strategy:  ```

## Compatibility and Testing

This tool has been tested and verified to work on the following environment:

- **Operating System**: Debian (Linux)

### Note:
While the tool is expected to work on other Unix-like systems (e.g., Ubuntu, Fedora), it has not been explicitly tested on these platforms. Testing on macOS and Windows is planned for future releases.

If you encounter any issues on other platforms, please report them in the Issues section.

## Contributing

Pull requests are welcome. For major changes, please open an issue first
to discuss what you would like to change.

Please make sure to update tests as appropriate.
