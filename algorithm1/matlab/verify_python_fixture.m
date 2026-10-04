function result = verify_python_fixture(filename)
%VERIFY_PYTHON_FIXTURE Compare native MATLAB and decomposed QCLAB with Python.
% Run the Python demo first; filename points to its fixture.mat.
    f = load(filename);
    n = double(f.n_pixels); dim = 2^n;
    result = algorithm1_qclab(f.image, f.kappa, f.beta, double(f.seed_zero_based)+1, ...
        'Sigma',f.sigma, 'SigmaDDOF',f.sigma_ddof, 'Shots',double(f.shots));
    assert(norm(result.model.H-f.H,'fro') < 1e-12);
    assert(norm(result.rho-f.rho_reference,'fro') < 1e-9);
    assert(max(abs(result.gamma_exact-f.gamma_reference)) < 1e-10);
    % Recreate the actual compiled Python gates in the existing MATLAB toolbox.
    % Qiskit system q[k] -> QCLAB q[N-1-k]; ancilla q[N+k] -> q[2N-1-k].
    circuit = qclab.QCircuit(2*n);
    for row = 1:size(f.gates,1)
        q = map_qubit(double(f.gates(row,2)),n);
        if f.gates(row,1) == 1
            theta = f.u_angles(row,1); phi = f.u_angles(row,2); lambda = f.u_angles(row,3);
            % U(theta,phi,lambda) = Phase(phi) RY(theta) Phase(lambda).
            circuit.push_back(qclab.qgates.Phase(q,lambda));
            circuit.push_back(qclab.qgates.RotationY(q,theta));
            circuit.push_back(qclab.qgates.Phase(q,phi));
        else
            assert(f.gates(row,1) == 2, 'Unknown gate encoding.');
            target = map_qubit(double(f.gates(row,3)),n);
            circuit.push_back(qclab.qgates.CNOT(q,target));
        end
    end
    sim = circuit.simulate(repmat('0',1,2*n));
    % Global phase cancels in rho; it need not be applied to the QCLAB state.
    A = reshape(sim.states(),dim,dim).';
    rho = A*A';
    assert(norm(rho-f.rho_circuit,'fro') < 1e-9);
    result.decomposed_circuit = circuit;
    fprintf('Native and decomposed QCLAB circuits match Python.\n');
end

function q = map_qubit(qiskit_index,n)
    if qiskit_index < n
        q = n-1-qiskit_index;
    else
        q = 3*n-1-qiskit_index;
    end
end
