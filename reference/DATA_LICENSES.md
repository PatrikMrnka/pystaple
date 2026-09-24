# Reference data: origin and licenses

The meshes (`mesh_*.mat`), MATLAB outputs (`reference.json`, `diag_femur.json`) and
MATLAB OpenSim models (`osim/`) in this folder were generated from the
`bone_datasets` distributed with [STAPLE](https://github.com/modenaxe/msk-STAPLE)
(`export_reference_outputs.m`, `hip_model.m`). They are used only to test pystaple
and are not included in the Python package.

Each dataset keeps the terms under which it is distributed. Please cite the
reference publication when using the data.

| Dataset | Source | Reference publication | Terms |
|---|---|---|---|
| ICL_MRI | STAPLE `bone_datasets` | Modenese & Renault (2021), [doi:10.1016/j.jbiomech.2020.110186](https://doi.org/10.1016/j.jbiomech.2020.110186) | as STAPLE (CC BY-NC 4.0); cite the STAPLE publication |
| JIA_MRI | STAPLE `bone_datasets` | Montefiori et al. (2019), Ann Biomed Eng 47:2155-2167, [doi:10.1007/s10439-019-02287-0](https://doi.org/10.1007/s10439-019-02287-0) | as STAPLE (CC BY-NC 4.0); cite the publication |
| LHDL_CT | Living Human Digital Library, via STAPLE | Viceconti et al. (2008), J Physiol Sci, [doi:10.2170/physiolsci.RP009908](https://doi.org/10.2170/physiolsci.RP009908) | **CC BY-NC-SA 2.0 BE**, see `LHDL_CT/LICENSE_LHDL.pdf` |
| MC22 | Montefiori et al. (2020) dataset, [doi:10.15131/shef.data.9934055.v1](https://doi.org/10.15131/shef.data.9934055.v1), via STAPLE | Montefiori et al. (2020), PLoS ONE 15:e0242973, [doi:10.1371/journal.pone.0242973](https://doi.org/10.1371/journal.pone.0242973) | terms of the original dataset; cite the publication |
| TLEM2_CT, TLEM2_MRI | TLEM 2.0 dataset, via STAPLE | Carbone et al. (2015), J Biomech 48:734-741, [doi:10.1016/j.jbiomech.2014.12.034](https://doi.org/10.1016/j.jbiomech.2014.12.034) | terms of the TLEM 2.0 dataset; cite the publication |
| VAKHUM_CT | VAKHUM project (Physiome Space), via STAPLE | Van Sint Jan (2006), [doi:10.1080/14639220412331529591](https://doi.org/10.1080/14639220412331529591) | **CC BY-NC-SA 2.0 BE**, see `VAKHUM_CT/VAKHUM_resources_license_agreement.pdf` |

The files in `LHDL_CT/` and `VAKHUM_CT/` are derivative works of data licensed
under CC BY-NC-SA 2.0 BE and are therefore distributed under that same license
(non-commercial use, attribution to the original authors, share alike):
http://creativecommons.org/licenses/by-nc-sa/2.0/be/
