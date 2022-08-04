function gradient_model =  compute_anatomicalgradient(varargin)

%USAGE:             gradient_model = compute_anatomicalgradient(Vertices,VertConn,nIterations,clin_data,meg_data,n_perms,covars)
%PURPOSE:           Compute anatomical gradient effects as in: "Aberrant neurophysiological signaling underlies speech impairments in Parkinson’s disease" (Wiesman et al., 2022)
%
%REQUIRED INPUTS:   Vertices: from Brainstorm - [Nvertices x 3], coordinates (x,y,z) of all the points of the surface, in SCS coordinates
%                   VertConn: from Brainstorm - [Nvertices x Nvertices] Sparse adjacency matrix, VertConn(i,j)=1 if i and j are neighbors
%                   nIterations: from Brainstorm - number of tess_smooth iterations - should be set to 46 for a 15k-vertex surface (for 100% smoothing: nIterations = ceil(300 * length(iVertices) / 100000))
%                   clin_data: [Npatients x 1] vector of clinical data (patient 1...n)
%                   meg_data: [Nvertices x Npatients] - source-imaged MEG data (e.g., slowing index; patient 1...n)
%
%OPTIONAL INPUTS:   n_perms: number of data permutations for null distribution generation (default: 1000)
%                   covars: nuisance covariates for computing partial correlations (default: no nuisance covariates, standard Pearson correlation)
%                   
%OUTPUTS:           gradient_model: structure containing gradient model unstandardized beta weight, normalized vertex-wise clinical correlation coefficients, permutation matrix [Npermutations x Npatients],  permuted gradient model unstandardized beta weights [Npermutations], and permuted two-tailed p-values
%
%AUTHOR:            Alex I. Wiesman, neuroSPEED lab, Montreal Neurological Institute
%VERSION HISTORY:   08/03/2022  v1: First working version of program
%
%LICENSE:           This software is distributed under the terms of the GNU General Public License as published by the Free Software Foundation. Further details on the GPLv3 license can be found at http://www.gnu.org/copyleft/gpl.html.
%                   FOR RESEARCH PURPOSES ONLY. THE SOFTWARE IS PROVIDED "AS IS," AND THE AUTHORS DO NOT MAKE ANY WARRANTY, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE, NOR DO THEY ASSUME ANY LIABILITY OR RESPONSIBILITY FOR THE USE OF THIS SOFTWARE.

%parse inputs and set defaults
if nargin < 5
    error('Not all required inputs have been provided.')
end

Vertices = varargin{1};
VertConn = varargin{2};
nIterations = varargin{3};
clin_data = varargin{4};
meg_data = varargin{5};

pearson_corr = false;
if nargin < 7
    pearson_corr = true;
else
    covars = varargin{7};
end

if nargin < 6
    n_perms = 1000;
else
    n_perms = varargin{6};
end

%smooth surface coordinates (reduces effects of gyrification on gradient estimation)
[Vertices, ~] = tess_smooth(Vertices, 1, nIterations,VertConn,0);

%compute vertex-wise clinical correlations
clin_corrs = nan(size(meg_data,1),1);
if pearson_corr
    for i = 1:size(meg_data,1)
        clin_corrs(i,1) = corr(clin_data,meg_data(i,:)');
    end
else
    for i = 1:size(meg_data,1)
        clin_corrs(i,1) = partialcorr(clin_data,meg_data(i,:)',covars);
    end
end

%normalize correlation coefficients 
clin_corrs_z = atanh(clin_corrs);
gradient_model.clin_corrs_z = clin_corrs_z;

%compute regression for anatomical gradient model (unstandardized beta)
gradient_model.beta = regress(clin_corrs_z,[ones(size(Vertices,1),1),Vertices]);   

%permutation testing
randomizations = nan(n_perms,size(meg_data,2));

for i = 1:n_perms
    fprintf('Running permutation #%d\n',i);
    clin_corrs_perm = nan(size(meg_data,1),1);
    randomizations(i,:) = randperm(size(meg_data,2));
    meg_data_rand = meg_data(:,randomizations(i,:)); 
    
    if pearson_corr
        for ii = 1:size(meg_data,1)
            clin_corrs_perm(ii,1) = corr(clin_data,meg_data_rand(ii,:)');
        end
    else
        for ii = 1:size(meg_data,1)
            clin_corrs_perm(ii,1) = partialcorr(clin_data,meg_data_rand(ii,:)',covars);
        end
    end
    
    clin_corrs_z_perm = atanh(clin_corrs_perm);
    [gradient_model.perm(:,i),~,~,~,~] = regress(clin_corrs_z_perm,[ones(size(Vertices,1),1),Vertices]);
end
gradient_model.randomizations = randomizations;

%calculate p-value
for i = 1:size(gradient_model.beta,1)
    if gradient_model.beta(i,1) > 0
        gradient_model.perm_pvalues(i,1) = (sum(gradient_model.perm(i,:) >= gradient_model.beta(i,1))/size(gradient_model.perm,2))*2;
    elseif gradient_model.beta(i,1) < 0
        gradient_model.perm_pvalues(i,1) = (sum(gradient_model.perm(i,:) <= gradient_model.beta(i,1))/size(gradient_model.perm,2))*2;
    end
end