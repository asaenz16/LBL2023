function info = setup_ising_petz()
%SETUP_ISING_PETZ Add the project-local QCLAB dependency for this session.
% Download once from the repository root:
% git clone --depth 1 --branch v1.2.0 https://github.com/QuantumComputingLab/qclab.git external/qclab
% No savepath or additional MathWorks toolbox is required.
    here = fileparts(mfilename('fullpath'));
    repo = fileparts(fileparts(here));
    dependency = fullfile(repo,'external','qclab');
    assert(isfile(fullfile(dependency,'+qclab','QCircuit.m')), ...
        'Download the official QCLAB v1.2.0 dependency into external/qclab. See help setup_ising_petz.');
    addpath(here); addpath(dependency);
    info = struct('matlab',version,'qclab_root',dependency, ...
        'qclab_release','v1.2.0','expected_commit','b9de8ca2d2e78186dc159a6ac6340646b7d603aa');
    head = fullfile(dependency,'.git','HEAD');
    if isfile(head), info.checkout_head = strtrim(fileread(head)); end
end
