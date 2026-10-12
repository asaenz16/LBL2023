function report = verify_ising_petz_qclab(fixture)
%VERIFY_ISING_PETZ_QCLAB Execute QCLAB circuits and compare independent Python fixtures.
    info=setup_ising_petz();here=fileparts(mfilename('fullpath'));root=fileparts(here);
    if nargin<1,fixture=fullfile(root,'results','ising-petz','qclab-fixture.mat');end
    f=load(fixture);model=ising_petz_qclab_model(f.image,'Alpha',f.alpha,'Threshold',f.threshold,'B',f.b);
    assert(abs(model.b-f.b)<1e-13);kraus_error=0;simulation_error=0;
    for i=1:model.n
        kernel=model.kernels{i};expected=f.kraus{i};
        for j=1:4,kraus_error=max(kraus_error,norm(full(kernel.kraus{j})-expected(:,:,j),'fro'));end
        % Execute every initialized input column through QCLAB.simulate.
        % Bath is the last qubit (LSB), so initialized columns have odd indices.
        for block={kernel.erase,kernel.recover}
            circuit=block{1};u=full(circuit.matrix());dim=size(u,1);
            for c=1:2:dim
                input=zeros(dim,1);input(c)=1;s=circuit.simulate(input);
                simulation_error=max(simulation_error,norm(s.states()-u(:,c)));
            end
        end
    end
    assert(kraus_error<1e-11,'QCLAB Kraus matrices differ from analytic Python operators.');
    assert(simulation_error<1e-11,'QCLAB state-vector execution differs from block matrices.');
    density=algorithm1_ising_petz_qclab(4,'Sweeps',50,'Simulation','density');
    population=algorithm1_ising_petz_qclab(4,'Sweeps',50,'Simulation','population');
    larger=algorithm1_ising_petz_qclab(8,'Sweeps',50,'Simulation','population');
    density_error=norm(density.rho_final-f.rho_after_50,'fro');
    history_error=max(abs(density.gamma-f.gamma_history),[],'all');
    reduction_error=max(abs(density.gamma-population.gamma),[],'all');
    larger_error=max(abs(larger.gamma-f.gamma_history_L8),[],'all');
    transfer_error=max([max(abs(density.gamma_gibbs-f.gamma_gibbs),[],'all'), ...
        max(abs(larger.gamma_gibbs-f.gamma_gibbs_L8),[],'all')]);
    assert(max([density_error,history_error,reduction_error,larger_error,transfer_error])<1e-10);
    trajectory=algorithm1_ising_petz_qclab(4,'Sweeps',20,'Simulation','trajectories','Shots',16384,'RngSeed',88720);
    target=f.gamma_history(1:21,:,:);error=abs(trajectory.gamma-target);
    assert(max(error,[],'all')<.015 && all(error<6*trajectory.gamma_se+.002,'all'));
    resources=algorithm1_ising_petz_qclab(64,'Simulation','none');
    assert(resources.model.gate_counts_per_sweep.cx==1520);
    assert(resources.model.gate_counts_per_sweep.ry==632);
    assert(numel(resources.model.layers)==4);
    report=struct('environment',info, ...
        'kraus_error',kraus_error,'qclab_simulate_error',simulation_error, ...
        'density_error_after_50_sweeps',density_error,'all_sweeps_correlation_error',history_error, ...
        'density_vs_population_error',reduction_error,'L8_population_history_error',larger_error, ...
        'transfer_reference_error',transfer_error,'matlab_trajectory_max_error',max(error,[],'all'), ...
        'N128_circuit_counts',resources.model.gate_counts_per_sweep, ...
        'density_simulation_spins',8,'exact_population_simulation_spins',16, ...
        'large_circuit_construction_spins',128,'all_checks_passed',true);
    file=fullfile(root,'ising-petz-matlab-validation.json');fid=fopen(file,'w');
    assert(fid>=0);clean=onCleanup(@()fclose(fid));fprintf(fid,'%s\n',jsonencode(report,PrettyPrint=true));
    disp(report);
end
