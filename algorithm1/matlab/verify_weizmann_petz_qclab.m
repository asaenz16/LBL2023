function report = verify_weizmann_petz_qclab()
%VERIFY_WEIZMANN_PETZ_QCLAB Native QCLAB checks on natural and frustrated images.
    environment=setup_ising_petz();here=fileparts(mfilename('fullpath'));root=fileparts(here);
    f=load(fullfile(root,'results','weizmann-petz','matlab-fixture.mat'));
    r=algorithm1_ising_image_qclab(f.image_small,'Seeds',f.seeds_small,'B',f.b_small, ...
        'Sweeps',20,'Simulation','density','GibbsReference',f.reference_small);
    density_error=norm(r.rho_final-f.rho_small,'fro');
    history_error=max(abs(r.gamma-f.history_small),[],'all');
    nine=algorithm1_ising_image_qclab(f.image_nine,'Seeds',f.seeds_nine,'B',f.b_nine, ...
        'Sweeps',2,'Simulation','density');
    nine_error=norm(nine.rho_final-f.rho_nine,'fro');
    reference_error=max(abs(nine.gamma_gibbs-f.reference_nine),[],'all');
    kraus_error=0;simulate_error=0;counts=struct;
    for item={'small','square8','square16'}
        name=item{1};image=f.(['image_' name]);b=f.(['b_' name]);
        model=ising_petz_qclab_model(image,'B',b);center=f.(['center_' name]);
        kernel=model.kernels{center};expected=f.(['kraus_' name]);
        for j=1:4,kraus_error=max(kraus_error,norm(full(kernel.kraus{j})-expected(:,:,j),'fro'));end
        for segment={kernel.erase,kernel.recover}
            circuit=segment{1};matrix=full(circuit.matrix());dim=size(matrix,1);
            for c=1:2:dim
                state=zeros(dim,1);state(c)=1;execution=circuit.simulate(state);
                simulate_error=max(simulate_error,norm(execution.states()-matrix(:,c)));
            end
        end
        counts.(name)=model.gate_counts_per_sweep;
    end
    sampled=algorithm1_ising_image_qclab(f.image_square8,'Seeds',f.seeds_square8,'B',f.b_square8, ...
        'Sweeps',20,'Shots',8192,'Simulation','trajectories','RngSeed',187163, ...
        'GibbsReference',f.reference_square8);
    difference=abs(squeeze(sampled.gamma(end,:,:))-f.gamma_square8_20);
    combined_se=sqrt(squeeze(sampled.gamma_se(end,:,:)).^2+f.se_square8_20.^2);
    assert(all(difference<6*combined_se+.002,'all'));
    assert(max([density_error,history_error,nine_error,reference_error,kraus_error,simulate_error])<1e-10);
    report=struct('environment',environment,'natural_8_spin_density_error',density_error, ...
        'natural_correlation_history_error',history_error,'frustrated_9_spin_density_error',nine_error, ...
        'nine_spin_enumeration_vs_transfer_error',reference_error,'local_kraus_error',kraus_error, ...
        'qclab_statevector_error',simulate_error,'native_64_spin_trajectory_max_difference',max(difference,[],'all'), ...
        'counts',counts,'all_checks_passed',true);
    file=fullfile(root,'weizmann-petz-matlab-validation.json');fid=fopen(file,'w');assert(fid>=0);
    cleanup=onCleanup(@()fclose(fid));fprintf(fid,'%s\n',jsonencode(report,PrettyPrint=true));disp(report);
end
