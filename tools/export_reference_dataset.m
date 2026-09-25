function export_reference_dataset(staple_dir, bones_folder, ref_dir, dataset, body_mass)
% EXPORT_REFERENCE_DATASET  MATLAB STAPLE reference for one new dataset.
%
%   export_reference_dataset(staple_dir, bones_folder, ref_dir, dataset)
%   export_reference_dataset(..., body_mass)          % default 64 kg
%
% Writes the same files as the existing reference datasets:
%   <ref_dir>/<dataset>/mesh_<bone>.mat   Points + ConnectivityList (1-based)
%   <ref_dir>/<dataset>/reference.json    CS, JCS, BL of STAPLE_pelvis,
%                                         GIBOC_femur (cylinder), Kai2014_tibia
%   <ref_dir>/<dataset>/osim/             bone_model.osim + Geometry/ (hip_model.m)
%
% bones_folder must contain pelvis_no_sacrum, femur_r and tibia_r (.stl or
% .mat), e.g. the bones_STAPLE folder of an msk-PIPE run:
%
%   export_reference_dataset('C:\msk-PIPE\src\STAPLE', ...
%       'C:\msk-PIPE\data\output\run_1\bones_STAPLE', ...
%       'C:\STAPLE_python\pystaple\reference', 'MSKPIPE_CT')
%
% staple_dir may be the msk-STAPLE repository or its STAPLE subfolder.
% The OpenSim MATLAB API must be configured (needed by hip_model.m).

if nargin < 5, body_mass = 64; end

staple_dir = char(java.io.File(staple_dir).getCanonicalPath());
if isfolder(fullfile(staple_dir, 'STAPLE'))
    staple_dir = fullfile(staple_dir, 'STAPLE');
end
addpath(genpath(staple_dir));

bones_list = {'pelvis_no_sacrum', 'femur_r', 'tibia_r'};
out_dir = fullfile(ref_dir, dataset);
osim_dir = fullfile(out_dir, 'osim');
if ~isfolder(osim_dir), mkdir(osim_dir); end

% --- input meshes, exactly as MATLAB reads them --------------------------------
% pystaple reads mesh_<bone>.mat, so both sides start from the same triangulation
% (same vertex order) and not from two different STL readers.
geom_set = createTriGeomSet(bones_list, bones_folder);
for n = 1:numel(bones_list)
    tri = geom_set.(bones_list{n});
    Points = tri.Points; %#ok<NASGU>
    ConnectivityList = tri.ConnectivityList; %#ok<NASGU>
    save(fullfile(out_dir, ['mesh_', bones_list{n}, '.mat']), 'Points', 'ConnectivityList');
end

% --- bone analyses -> reference.json -------------------------------------------
ref = struct();
ref.dataset = dataset;
ref.algorithms = struct('pelvis', 'STAPLE', 'femur', 'GIBOC-cylinder', 'tibia', 'Kai2014');
ref.matlab_version = version;
ref.JCS = struct(); ref.BL = struct(); ref.CS = struct(); ref.errors = struct();

% {input mesh, key in reference.json, algorithm}
jobs = {
    'pelvis_no_sacrum', 'pelvis',  @(t) STAPLE_pelvis(t, 'r', 0, 0, 1)
    'femur_r',          'femur_r', @(t) GIBOC_femur(t, 'r', 'cylinder', 0, 0, 1)
    'tibia_r',          'tibia_r', @(t) Kai2014_tibia(t, 'r', 0, 0, 1)
};
for n = 1:size(jobs, 1)
    [bone, key, algo] = jobs{n, :};
    fprintf('[%s] %s ... ', dataset, bone);
    rng(0);
    try
        [CS, JCS, BL] = algo(geom_set.(bone));
        ref.CS.(key) = numeric_only(CS);
        ref.JCS.(key) = numeric_only(JCS);
        ref.BL.(key) = numeric_only(BL);
        fprintf('ok\n');
    catch err
        ref.errors.(bone) = err.message;
        fprintf('FAILED: %s\n', err.message);
    end
    close all;
end
fid = fopen(fullfile(out_dir, 'reference.json'), 'w', 'n', 'UTF-8');
fwrite(fid, jsonencode(ref, 'PrettyPrint', true), 'char');
fclose(fid);

% --- OpenSim model, as hip_model.m -----------------------------------------------
fprintf('[%s] hip_model ... ', dataset);
cur_model_name = 'auto2020_hip_R';
writeModelGeometriesFolder(geom_set, fullfile(osim_dir, 'Geometry'), 'obj');
rng(0);
[JCS, BL] = processTriGeomBoneSet(geom_set);
close all;
osimModel = initializeOpenSimModel(cur_model_name);
osimModel = addBodiesFromTriGeomBoneSet(osimModel, geom_set, 'Geometry', 'obj');
createOpenSimModelJoints(osimModel, JCS, 'auto2020');
osimModel = assignMassPropsToSegments(osimModel, JCS, body_mass);
addBoneLandmarksAsMarkers(osimModel, BL);
osimModel.finalizeConnections();
osimModel.print(fullfile(osim_dir, 'bone_model.osim'));
fprintf('ok\n');

rmpath(genpath(staple_dir));
fprintf('reference written to %s\n', out_dir);
end


function out = numeric_only(s)
% Numeric fields of a (nested) struct; triangulations etc. are dropped.
out = struct();
if ~isstruct(s), return; end
for f = fieldnames(s)'
    v = s.(f{1});
    if isstruct(v) && isscalar(v)
        out.(f{1}) = numeric_only(v);
    elseif (isnumeric(v) || islogical(v)) && ~isempty(v)
        out.(f{1}) = double(v);
    end
end
end