function result = algorithm1_qclab(image, kappa, beta, seed, varargin)
%ALGORITHM1_QCLAB Exact Gibbs purification and seed correlations in QCLAB.
% seed is ONE-BASED row-major pixel index. beta is the manuscript parameter.
% Requires QCLAB with MatrixGate (v1.1.0+). This is an exponential reference.
% Native circuit uses two multi-qubit matrix blocks and N CNOTs. See
% verify_python_fixture for the equivalent decomposed one/two-qubit circuit.
    p = inputParser;
    addParameter(p, 'Sigma', []);
    addParameter(p, 'SigmaDDOF', 1);
    addParameter(p, 'Shots', 8192, @(x) isscalar(x) && isfinite(x) && x >= 1 && x == fix(x));
    addParameter(p, 'RngSeed', 1200);
    parse(p, varargin{:});
    validateattributes(beta, {'numeric'}, {'real','finite','scalar','nonnegative'});
    model = algorithm1_model(image, kappa, 'Sigma',p.Results.Sigma, ...
        'SigmaDDOF',p.Results.SigmaDDOF);
    n = model.n; dim = 2^n;
    validateattributes(seed, {'numeric'}, {'scalar','integer','>=',1,'<=',n});
    assert(exist('qclab.qgates.MatrixGate','class') == 8, ...
        'Add the QCLAB toolbox root to the MATLAB path (MatrixGate required).');
    [U,D] = eig(model.H);
    [energies, order] = sort(real(diag(D)));
    U = U(:,order);
    weights = exp(-(beta/n)*(energies-energies(1)));
    weights = weights / sum(weights);
    amplitude = sqrt(weights);
    % Real Householder preparation: P|0> = sqrt(weights).
    v = -amplitude; v(1) = v(1) + 1;
    if norm(v) == 0
        P = eye(dim);
    else
        v = v / norm(v);
        P = eye(dim) - 2*(v*v');
    end
    circuit = qclab.QCircuit(2*n);
    % QCLAB qubit 0 is the MSB: system 0:N-1, ancilla N:2N-1.
    circuit.push_back(qclab.qgates.MatrixGate(n:2*n-1, P, 'sqrt(p)'));
    for k = 0:n-1
        circuit.push_back(qclab.qgates.CNOT(n+k, k));
    end
    circuit.push_back(qclab.qgates.MatrixGate(0:n-1, U, 'energy basis U'));
    simulation = circuit.simulate(repmat('0',1,2*n));
    state = simulation.states();
    % A(system,ancilla) for |system> tensor |ancilla>; nonconjugating transpose.
    A = reshape(state, dim, dim).';
    rho = A*A';
    rho_reference = expm(-(beta/n)*(model.H - energies(1)*eye(dim)));
    rho_reference = rho_reference/trace(rho_reference);
    trace_distance = sum(svd(rho-rho_reference))/2;
    assert(trace_distance < 1e-9, 'QCLAB circuit disagrees with expm reference.');
    z = zeros(dim,n);
    for site = 1:n
        z(:,site) = 1 - 2*double(bitget(uint64((0:dim-1)'), n-site+1));
    end
    observables = z.*z(:,seed)/4;
    gamma_exact = real(diag(rho_reference)).' * observables;
    gamma_circuit = real(diag(rho)).' * observables;
    % Repeated ideal computational-basis measurements of the simulated state.
    previous_rng = rng;
    cleanup = onCleanup(@() rng(previous_rng)); %#ok<NASGU>
    rng(p.Results.RngSeed, 'twister');
    cdf = cumsum(max(real(diag(rho)),0)); cdf = cdf/cdf(end); cdf(end) = 1;
    counts = zeros(dim,1);
    for shot = 1:p.Results.Shots
        outcome = find(rand < cdf, 1);
        counts(outcome) = counts(outcome) + 1;
    end
    gamma_shots = counts.' * observables / p.Results.Shots;
    radius = sqrt(log(2*n/0.05)/(8*p.Results.Shots));
    lower = max(-0.25,gamma_shots-radius); upper = min(0.25,gamma_shots+radius);
    lower(seed) = 0.25; upper(seed) = 0.25;
    result = struct('model',model, 'circuit',circuit, 'rho',rho, ...
        'rho_reference',rho_reference, 'beta_effective',beta/n, ...
        'energies',energies, 'weights',weights, 'trace_distance',trace_distance, ...
        'gamma_exact',gamma_exact, 'gamma_circuit',gamma_circuit, ...
        'gamma_shots',gamma_shots, 'counts',counts, 'ci_lower',lower, 'ci_upper',upper, ...
        'sign_resolved',(lower>0 | upper<0), ...
        'same_segment',reshape(gamma_shots>0,size(image,2),size(image,1)).');
end
