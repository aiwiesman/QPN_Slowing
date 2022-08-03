%PURPOSE:           Preprocess, source image, and compute PSDs from MEG data as in: "A sagittal gradient of pathological and compensatory effects of neurophysiological slowing in Parkinson’s disease" (Wiesman et al., 2022)
%
%NOTES:             At each stage, the preprocessed files from the previous step should be input as the cell array sFiles (unless otherwise specified)
%                   This pipeline is NOT fully automated - numerous steps are non-contiguous with those before/after, meaning manual inspection/intervention is required before proceeding (see notes throughout this code)
%
%AUTHOR:            Alex I. Wiesman, neuroSPEED lab, Montreal Neurological Institute
%VERSION HISTORY:   08/03/2022  v1: First working version of script
%
%LICENSE:           This software is distributed under the terms of the GNU General Public License as published by the Free Software Foundation. Further details on the GPLv3 license can be found at http://www.gnu.org/copyleft/gpl.html.
%                   FOR RESEARCH PURPOSES ONLY. THE SOFTWARE IS PROVIDED "AS IS," AND THE AUTHORS DO NOT MAKE ANY WARRANTY, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE, NOR DO THEY ASSUME ANY LIABILITY OR RESPONSIBILITY FOR THE USE OF THIS SOFTWARE.

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


%%%Create database structure and import MEG data prior to preprocessing


%%%%%%%%%%%%%%%%%%%%%%%%%%PREPROCESSING%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%%
%%%Sensor-level PSD computation - use to exclude bad channels (clear outliers in PSD)
%%%

% Input files
sFiles = {};

% Start a new report
bst_report('Start', sFiles);

% Process: Power spectrum density (Welch)
sFiles = bst_process('CallProcess', 'process_psd', sFiles, [], ...
    'timewindow',  [], ...
    'win_length',  3, ...
    'win_overlap', 50, ...
    'units',       'physical', ...  % Physical: U2/Hz
    'sensortypes', 'MEG', ...
    'win_std',     0, ...
    'edit',        struct(...
         'Comment',         'Power', ...
         'TimeBands',       [], ...
         'Freqs',           [], ...
         'ClusterFuncTime', 'none', ...
         'Measure',         'power', ...
         'Output',          'all', ...
         'SaveKernel',      0));

% Save and display report
ReportFile = bst_report('Save', sFiles);
bst_report('Open', ReportFile);


%%%
%%%Filtering and SSP detection (ECG & EOG)
%%%

% Input files
sFiles = {};

% Start a new report
bst_report('Start', sFiles);

% Process: Band-pass:1Hz-200Hz
sFiles = bst_process('CallProcess', 'process_bandpass', sFiles, [], ...
    'sensortypes', 'MEG', ...
    'highpass',    1, ...
    'lowpass',     200, ...
    'tranband',    0, ...
    'attenuation', 'strict', ...  % 60dB
    'ver',         '2019', ...  % 2019
    'mirror',      0, ...
    'read_all',    1);

% Process: Notch filter: 60Hz 120Hz 180Hz
sFiles = bst_process('CallProcess', 'process_notch', sFiles, [], ...
    'sensortypes', 'MEG', ...
    'freqlist',    [60, 120, 180], ...
    'cutoffW',     1, ...
    'useold',      1, ...
    'read_all',    1);

% Process: Detect heartbeats
sFiles = bst_process('CallProcess', 'process_evt_detect_ecg', sFiles, [], ...
    'channelname', 'ECG', ...
    'timewindow',  [], ...
    'eventname',   'cardiac');

% Process: Detect eye blinks
sFiles = bst_process('CallProcess', 'process_evt_detect_eog', sFiles, [], ...
    'channelname', 'VEOG, HEOG', ...
    'timewindow',  [], ...
    'eventname',   'blink');

% Save and display report
ReportFile = bst_report('Save', sFiles);
bst_report('Open', ReportFile);


%%%At this stage, ECG and EOG artifacts should be manually inspected (and adjusted, if necessary)
%%%Individualized SSPs should also be applied to correct highly-stereotyped artifacts on a case-by-case basis


%%%
%%%Compute SSPs
%%%

% Input files
sFiles = {};

% Start a new report
bst_report('Start', sFiles);

% Process: Remove simultaneous
sFiles = bst_process('CallProcess', 'process_evt_remove_simult', sFiles, [], ...
    'remove', 'cardiac', ...
    'target', 'blink', ...
    'dt',     0.25, ...
    'rename', 0);

% Process: SSP ECG: cardiac
sFiles = bst_process('CallProcess', 'process_ssp_ecg', sFiles, [], ...
    'eventname',   'cardiac', ...
    'sensortypes', 'MEG', ...
    'usessp',      1, ...
    'select',      1);

% Process: SSP EOG: blink
sFiles = bst_process('CallProcess', 'process_ssp_eog', sFiles, [], ...
    'eventname',   'blink', ...
    'sensortypes', 'MEG', ...
    'usessp',      1, ...
    'select',      1);

% Save and display report
ReportFile = bst_report('Save', sFiles);
bst_report('Open', ReportFile);


%%%At this stage, ECG & EOG SSP components should be manually inspected and accepted/rejected based on spatial and temporal topographies


%%%
%%%Resample continuous data and import as 6 second epochs
%%%

% Input files
sFiles = {};
SubjectNames = bst_gen_subnames(sFiles);

for i = 1:size(sFiles,2)

% Start a new report
bst_report('Start', sFiles{i});

% Process: Resample: 600Hz
sFiles = bst_process('CallProcess', 'process_resample', sFiles, [], ...
    'freq',     600, ...
    'read_all', 1);

% Process: Import MEG/EEG: Time
sFiles = bst_process('CallProcess', 'process_import_data_time', sFiles, [], ...
    'subjectname', SubjectNames{i}, ...
    'condition',   '', ...
    'timewindow',  [], ...
    'split',       6, ...
    'ignoreshort', 1, ...
    'usectfcomp',  1, ...
    'usessp',      1, ...
    'freq',        [], ...
    'baseline',    []);

% Save and display report
ReportFile = bst_report('Save', sFiles{i});
bst_report('Open', ReportFile);

end


%%%At this point, the 6 second epochs containing outlier values in terms of peak-to-peak amplitude or maximum signal gradient should be excluded using the ArtifactScanTool
%%%found at: https://github.com/nichrishayes/ArtifactScanTool -- or alternatively, for this analysis a union of two empirical thresholds can be applied (exclude epochs with 
%%%amplitude or gradient values > 3 median absolute deviations from the median across epochs within each recording session)

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


%%%Import Freesurfer-segmented anatomy folders at this point
%%%Quality of sMRI segmentation and surface reconstructions should be inspected


%%%%%%%%%%%%%%%%%%%%%%%%%%SOURCE-IMAGING%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%%
%%%Automated refinement of MEG-sMRI coregistration
%%%

% Input files
sFiles = {};

% Start a new report
bst_report('Start', sFiles);

% Process: Refine registration
sFiles = bst_process('CallProcess', 'process_headpoints_refine', sFiles, [], ...
    'tolerance', 0);

% Save and display report
ReportFile = bst_report('Save', sFiles);
bst_report('Open', ReportFile);


%%%At this stage, anatomical coregistration of the MEG data should be carefully inspected and corrected manually, where necessary

%%%
%%%Empty room preprocessing and noise covariance estimation
%%%

% Input files - here, sFiles should point to the empty room recordings
sFiles = {};

% Start a new report
bst_report('Start', sFiles);

% Process: Band-pass:1Hz-200Hz
sFiles = bst_process('CallProcess', 'process_bandpass', sFiles, [], ...
    'sensortypes', 'MEG', ...
    'highpass',    1, ...
    'lowpass',     200, ...
    'tranband',    0, ...
    'attenuation', 'strict', ...  % 60dB
    'ver',         '2019', ...  % 2019
    'mirror',      0, ...
    'read_all',    1);

% Process: Notch filter: 60Hz 120Hz 180Hz
sFiles = bst_process('CallProcess', 'process_notch', sFiles, [], ...
    'sensortypes', 'MEG', ...
    'freqlist',    [60, 120, 180], ...
    'cutoffW',     1, ...
    'useold',      1, ...
    'read_all',    1);

% Process: Resample: 600Hz
sFiles = bst_process('CallProcess', 'process_resample', sFiles, [], ...
    'freq',     600, ...
    'read_all', 1);

% Process: Compute covariance (noise or data)
sFiles = bst_process('CallProcess', 'process_noisecov', sFiles, [], ...
    'baseline',       [], ...
    'datatimewindow', [], ...
    'sensortypes',    'MEG', ...
    'target',         1, ...  % Noise covariance     (covariance over baseline time window)
    'dcoffset',       1, ...  % Block by block, to avoid effects of slow shifts in data
    'identity',       0, ...
    'copycond',       0, ...
    'copysubj',       0, ...
    'copymatch',      1, ...
    'replacefile',    1);  % Replace

% Save and display report
ReportFile = bst_report('Save', sFiles);
bst_report('Open', ReportFile);


%%%
%%%Source imaging using dSPM
%%%

% Input files - return to using participant MEG data for sFiles
sFiles = {};

% Start a new report
bst_report('Start', sFiles);

% Process: Compute head model
sFiles = bst_process('CallProcess', 'process_headmodel', sFiles, [], ...
    'Comment',     '', ...
    'sourcespace', 1, ...  % Cortex surface
    'meg',         3, ...  % Overlapping spheres
    'eeg',         3, ...  % OpenMEEG BEM
    'ecog',        2, ...  % OpenMEEG BEM
    'seeg',        2, ...  % OpenMEEG BEM
    'openmeeg',    struct(...
         'BemFiles',     {{}}, ...
         'BemNames',     {{'Scalp', 'Skull', 'Brain'}}, ...
         'BemCond',      [1, 0.0125, 1], ...
         'BemSelect',    [1, 1, 1], ...
         'isAdjoint',    0, ...
         'isAdaptative', 1, ...
         'isSplit',      0, ...
         'SplitLength',  4000), ...
    'channelfile', '');

% Process: Compute covariance (noise or data)
sFiles = bst_process('CallProcess', 'process_noisecov', sFiles, [], ...
    'baseline',       [], ...
    'datatimewindow', [], ...
    'sensortypes',    'MEG', ...
    'target',         2, ...  % Data covariance      (covariance over data time window)
    'dcoffset',       1, ...  % Block by block, to avoid effects of slow shifts in data
    'identity',       0, ...
    'copycond',       0, ...
    'copysubj',       0, ...
    'copymatch',      0, ...
    'replacefile',    1);  % Replace

% Process: Compute sources [2018]
sFiles = bst_process('CallProcess', 'process_inverse_2018', sFiles, [], ...
    'output',  1, ...  % Kernel only: shared
    'inverse', struct(...
         'Comment',        'dSPM-unscaled: MEG', ...
         'InverseMethod',  'minnorm', ...
         'InverseMeasure', 'dspm2018', ...
         'SourceOrient',   {{'free'}}, ...
         'Loose',          0.2, ...
         'UseDepth',       1, ...
         'WeightExp',      0.5, ...
         'WeightLimit',    10, ...
         'NoiseMethod',    'reg', ...
         'NoiseReg',       0.1, ...
         'SnrMethod',      'fixed', ...
         'SnrRms',         1e-06, ...
         'SnrFixed',       3, ...
         'ComputeKernel',  1, ...
         'DataTypes',      {{'MEG'}}));

% Save and display report
ReportFile = bst_report('Save', sFiles);
bst_report('Open', ReportFile);


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%



%%%%%%%%%%%%%%%%%%%%%%%%%SOURCE-LEVEL PSD%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Input files - this should be a list of ALL epoch-wise source-imaged data
sFiles = {};

sub_names = bst_gen_subnames(sFiles);
sFiles = bst_sep_files_by_subnames(sFiles,sub_names);

for i = 1:size(sFiles,1)

% Start a new report
bst_report('Start', sFiles{i});

% Process: Power spectrum density (Welch)
sFiles{i} = bst_process('CallProcess', 'process_psd', sFiles{i}, [], ...
    'timewindow',  [], ...
    'win_length',  3, ...
    'win_overlap', 50, ...
    'units',       'physical', ...  % Physical: U2/Hz
    'clusters',    {}, ...
    'scoutfunc',   1, ...  % Mean
    'win_std',     0, ...
    'edit',        struct(...
         'Comment',         'Avg,Power', ...
         'TimeBands',       [], ...
         'Freqs',           [], ...
         'ClusterFuncTime', 'none', ...
         'Measure',         'power', ...
         'Output',          'average', ...
         'SaveKernel',      0));

% Process: Spectrum normalization
sFiles{i} = bst_process('CallProcess', 'process_tf_norm', sFiles{i}, [], ...
    'normalize', 'relative2020', ...  % Relative power (divide by total power)
    'overwrite', 0);

% Process: Project on default anatomy: surface
sFiles{i} = bst_process('CallProcess', 'process_project_sources', sFiles{i}, [], ...
    'headmodeltype', 'surface');  % Cortex surface

% Save and display report
ReportFile = bst_report('Save', sFiles{i});
bst_report('Open', ReportFile);

end


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


%%%From here, the source-imaged PSD data can be used for
%%%spectral parameterization with specparam/FOOOF, averaged over conventional
%%%frequency bands and used for computation of neural slowing
%%%(see function 'compute_neuralslowing.m'), etc.


%%%%%%%%%%%%%%%%%%%%%%%%%AUXILLARY FUNCTIONS%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function SubjectNames = bst_gen_subnames(files)

SubjectNames = cell(1,size(files,2));
for i = 1:size(files,2)
    split_name = split(files{i},'/');
    SubjectNames{i} = split_name{1};
end
SubjectNames = unique(SubjectNames);

end


function sep_files = bst_sep_files_by_subnames(sFiles,sub_names)

sep_files = cell(size(sub_names,2),1);
for i = 1:size(sFiles,2)
    for ii = 1:size(sub_names,2)
        split_name = split(sFiles{1,i},'/');
        if strcmp(split_name{1},sub_names{1,ii})
            sep_files{ii}{i} = sFiles{1,i};
        end
    end
end

for i = 1:size(sep_files,1)
    sep_files{i} = sep_files{i}(~cellfun(@isempty, sep_files{i}));
end
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%