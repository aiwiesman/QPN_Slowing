function neural_slowing = compute_neuralslowing(patient_data,control_data,freqs)

%PURPOSE:           Compute neural slowing as in: "A sagittal gradient of pathological and compensatory effects of neurophysiological slowing in Parkinson’s disease" (Wiesman et al., 2022)
%
%REQUIRED INPUTS:   patient_data: array of data for patient participants - dimensions: locations (e.g., vertices or ROIs) x participants x frequency samples (averaged over relevant frequency bands, if applicable)
%                   control_data: array of data for control/reference participants - dimensions: locations x participants x frequency samples
%                   freqs: center frequencies (in Hz) of frequency samples (1xN vector; for fitting the linear slope of z per Hz model)
%
%OUTPUTS:           neural_slowing: neural slowing indices (i.e., slopes) for each patient per location - dimensions: location x patient
%
%
%AUTHOR:            Alex I. Wiesman, neuroSPEED lab, Montreal Neurological Institute
%VERSION HISTORY:   08/03/2022  v1: First working version of program
%
%LICENSE:           This software is distributed under the terms of the GNU General Public License as published by the Free Software Foundation. Further details on the GPLv3 license can be found at http://www.gnu.org/copyleft/gpl.html.
%                   FOR RESEARCH PURPOSES ONLY. THE SOFTWARE IS PROVIDED "AS IS," AND THE AUTHORS DO NOT MAKE ANY WARRANTY, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE, NOR DO THEY ASSUME ANY LIABILITY OR RESPONSIBILITY FOR THE USE OF THIS SOFTWARE.

%check if input dimensions are equivalent
if ~isequal(size(patient_data,1),size(control_data,1)) || ~isequal(size(patient_data,3),size(control_data,3))
    error('Patient and control data matrices have unequal dimensions.')
elseif ~isequal(size(patient_data,3),size(freqs,2))
    error('Patient/control data matrices have a different number of frequency samples than "freqs".')
end

%compute PD zscores per freq
control_means = squeeze(mean(control_data,2));
control_sds = squeeze(std(control_data,[],2));

%compute normalized (z-scored) data per location/patient/frequency
for i = 1:size(patient_data,2)
    patient_zscores(:,i,:) = (squeeze(patient_data(:,i,:)) - control_means)./control_sds;
end

%compute neural slowing slopes
for i = 1:size(patient_data,1)
    for ii = 1:size(patient_data,2)
        fit_slope = polyfit(freqs,patient_zscores(i,ii,:),1);
        neural_slowing(i,ii) = fit_slope(1,1);
    end
end