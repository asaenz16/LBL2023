function result = verify_thermal_bath_fixture(filename, use_qclab)
%VERIFY_THERMAL_BATH_FIXTURE Reconstruct and replay Python thermal contacts.
% A fresh bath qubit is thermalized before every contact, then traced out.
% Pass true as second argument to build the contact unitaries through QCLAB.
% This function has not been executed in MATLAB in the development workspace.
    if nargin < 2, use_qclab = false; end
    f = load(filename);
    n = double(f.n_pixels); d = 2^n;
    model = algorithm1_model(f.image, f.kappa, 'Sigma',f.sigma, 'SigmaDDOF',f.sigma_ddof);
    assert(norm(model.H-f.H,'fro') < 1e-12);
    rho_target = expm(-(f.beta/n)*(model.H-min(eig(model.H))*eye(d)));
    rho_target = rho_target/trace(rho_target);
    assert(norm(rho_target-f.rho_gibbs,'fro') < 1e-10);
    contacts = size(f.unitaries_bath_first,3);
    operators = cell(contacts,4);
    permutations = reshape([1:d;d+1:2*d],[],1); % bath-first to system-first
    for c = 1:contacts
        a = f.lowering(:,:,c); omega = f.omega(c);
        p1 = exp(-f.beta_effective*omega)/(1+exp(-f.beta_effective*omega));
        assert(abs(p1-f.p_excited(c)) < 1e-12);
        V = [zeros(d),a';a,zeros(d)];
        angle = sqrt(2*f.dt);
        if omega == 0, angle = sqrt(f.dt); end
        U = expm(-1i*angle*V);
        assert(norm(U-f.unitaries_bath_first(:,:,c),'fro') < 1e-10);
        if use_qclab
            circuit = qclab.QCircuit(n+1);
            circuit.push_back(qclab.qgates.MatrixGate(0:n,U(permutations,permutations),'bath exchange'));
            U_system_first = circuit.matrix();
            [~,inverse] = sort(permutations);
            U = U_system_first(inverse,inverse);
        end
        probabilities = [1-p1,p1];
        for in = 0:1
            for out = 0:1
                operators{c,2*in+out+1} = sqrt(probabilities(in+1))*U(out*d+(1:d),in*d+(1:d));
            end
        end
        complete = zeros(d); invariant = zeros(d);
        for k = 1:4
            K = operators{c,k};
            complete = complete + K'*K;
            invariant = invariant + K*rho_target*K';
        end
        assert(norm(complete-eye(d),'fro') < 1e-10);
        assert(norm(invariant-rho_target,'fro') < 1e-10);
    end
    rho = f.rho_initial;
    for sweep = 1:double(f.sweeps)
        for c = 1:contacts
            next = zeros(d);
            for k = 1:4
                K = operators{c,k}; next = next + K*rho*K';
            end
            rho = next;
        end
    end
    assert(norm(rho-f.rho_final,'fro') < 1e-9);
    result = struct('rho',rho,'rho_target',rho_target, ...
        'trace_distance',sum(svd(rho-rho_target))/2,'contacts_per_sweep',contacts);
    fprintf('MATLAB bath replay matches Python; trace distance %.3e.\n',result.trace_distance);
end
