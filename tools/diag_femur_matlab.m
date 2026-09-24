function diag_femur_matlab(staple_dir, ref_dir, dataset)
% DIAG_FEMUR_MATLAB  Mezivysledky GIBOC_femur pro porovnani s Python portem.
%
%   diag_femur_matlab('msk-STAPLE', 'C:\STAPLE_python\pystaple\reference', 'JIA_MRI')
%
% Ulozi <ref_dir>/<dataset>/diag_femur.json. Postup je stejny jako v
% GIBOC_femur.m (vcetne rng(0) jako v export_reference_outputs.m).

staple_dir = char(java.io.File(staple_dir).getCanonicalPath());
addpath(genpath(fullfile(staple_dir, 'STAPLE')));

% privatni funkce z algorithms/private nejsou zvenku dostupne -> kopie do tempdir
tmp = fullfile(tempdir, 'staple_private_copy');
if ~isfolder(tmp), mkdir(tmp); end
copyfile(fullfile(staple_dir, 'STAPLE', 'algorithms', 'private', '*.m'), tmp);
addpath(tmp);
cleaner = onCleanup(@() rmpath(tmp));

S = load(fullfile(ref_dir, dataset, 'mesh_femur_r.mat'));
femurTri = triangulation(S.ConnectivityList, S.Points);

rng(0);
U = femur_guess_CS(femurTri, 0);
[ProxFemTri, DistFemTri] = cutLongBoneMesh(femurTri, U);
CoeffMorpho = computeTriCoeffMorpho(femurTri);
[V_all, CenterVol] = TriInertiaPpties(femurTri);
aux = struct('CenterVol', CenterVol, 'V_all', V_all);
Z0 = V_all(:,1);
Z0 = sign((mean(ProxFemTri.Points)-mean(DistFemTri.Points))*Z0)*Z0;
aux.Z0 = Z0;
aux = GIBOC_femur_fitSphere2FemHead(ProxFemTri, aux, CoeffMorpho, 0);
aux.X0 = cross(aux.Y0, aux.Z0);

[Areas, Alt] = TriSliceObjAlongAxis(DistFemTri, Z0, 1);
[~, Zepi] = fitCSA(Alt, Areas);
EpiFemTri = GIBOC_isolate_epiphysis(DistFemTri, Z0, 'distal');

% --- convex hull and condyle axes (GIBOC_femur_processEpiPhysis) ---
K = convhull(EpiFemTri.Points, 'simplify', false);
Ks = convhull(EpiFemTri.Points);
[IdxPointsPair, EdgesLength] = LargestEdgeConvHull(EpiFemTri.Points);
[IdCdlPts, U_Axes] = GIBOC_femur_processEpiPhysis(EpiFemTri, aux, aux.V_all, 0.5, 0.75);
hull = struct();
hull.n_tri = size(K, 1);
hull.n_vertices = numel(unique(K));
hull.simplified_n_tri = size(Ks, 1);
hull.simplified_n_vertices = numel(unique(Ks));
n_top = min(300, size(IdxPointsPair, 1));
hull.top_pairs = IdxPointsPair(1:n_top, :);   % 1-based
hull.top_lengths = EdgesLength(1:n_top);
hull.IdCdlPts = IdCdlPts;                     % 1-based
hull.U_Axes_sum = sum(U_Axes, 1);

[~, ~, aux] = GIBOC_femur_ArticSurf(EpiFemTri, aux, CoeffMorpho, 'full_condyles', 0);
[postMed, postLat, aux] = GIBOC_femur_ArticSurf(EpiFemTri, aux, CoeffMorpho, 'post_condyles', 0);
[aux, JCS] = CS_femur_CylinderOnCondyles(postLat, postMed, aux, 'r');

d = struct();
d.Zepi = Zepi;
d.slice_alt = Alt;
d.slice_areas = Areas;
d.epi_n_points = size(EpiFemTri.Points, 1);
d.epi_n_faces = size(EpiFemTri.ConnectivityList, 1);
d.X1 = aux.X1; d.Y1 = aux.Y1; d.PtNotch = aux.BL.PtNotch;
d.post_lat_n_points = size(postLat.Points, 1);
d.post_lat_n_faces = size(postLat.ConnectivityList, 1);
d.post_lat_points_mean = mean(postLat.Points);
d.post_med_n_points = size(postMed.Points, 1);
d.post_med_n_faces = size(postMed.ConnectivityList, 1);
d.post_med_points_mean = mean(postMed.Points);
% sphere centres are not stored in aux by the cylinder method -> recompute
CSSph = CS_femur_SpheresOnCondyles(postLat, postMed, aux, 'r');
d.sphere_center_lat = CSSph.sphere_center_lat;
d.sphere_center_med = CSSph.sphere_center_med;
d.Cyl_Pt = aux.Cyl_Pt; d.Cyl_Y = aux.Cyl_Y; d.Cyl_Radius = aux.Cyl_Radius;
d.knee_origin = JCS.knee_r.Origin;
d.hull = hull;

out = fullfile(ref_dir, dataset, 'diag_femur.json');
fid = fopen(out, 'w'); assert(fid > 0, 'Nelze zapsat %s', out);
fprintf(fid, '%s', jsonencode(d)); fclose(fid);
fprintf('Ulozeno: %s\n', out);
end
