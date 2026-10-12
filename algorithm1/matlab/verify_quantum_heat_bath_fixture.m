function result = verify_quantum_heat_bath_fixture(filename, use_qclab)
%VERIFY_QUANTUM_HEAT_BATH_FIXTURE Replay the physical XXZ Petz heat bath.
% Python exports row-major dense matrices, so no bit reversal is needed.
% With use_qclab=true, rebuild the four-spin elementary recovery circuits.
% QCLAB provides the unitary blocks; resets/partial traces are averaged here.
% This helper has not been executed in MATLAB in the development workspace.
    if nargin < 2, use_qclab = false; end
    f = load(filename);
    n = double(f.n_pixels); d = 2^n; half = d/2;
    H = reconstruct_hamiltonian(f.image, f.kappa, f.sigma);
    assert(norm(H-f.H,'fro') < 1e-10, 'Image Hamiltonian mismatch.');
    target = expm(-double(f.b)*(H-min(eig(H))*eye(d)));
    target = target/trace(target);
    assert(norm(target-f.rho_gibbs,'fro') < 1e-10);
    assert(abs(double(f.beta)/n-double(f.b)) < 1e-12);
    circuits = cell(1,n); operators = cell(1,n);
    for site = 1:n
        K = f.kraus{site};
        complete = zeros(d); invariant = zeros(d);
        for k = 1:size(K,3)
            complete = complete + K(:,:,k)'*K(:,:,k);
            invariant = invariant + K(:,:,k)*target*K(:,:,k)';
        end
        assert(norm(complete-eye(d),'fro') < 1e-9);
        assert(norm(invariant-target,'fro') < 1e-9);
        operators{site} = K;
        if use_qclab
            gates = f.recovery_gates{site};
            assert(~isempty(gates), 'QCLAB gate replay is exported for four spins only.');
            circuit = qclab.QCircuit(n+1);
            for row = 1:size(gates,1)
                % Recovery uses a contiguous little-endian Qiskit register.
                q = n-double(gates(row,2));
                if gates(row,1) == 1
                    circuit.push_back(qclab.qgates.Phase(q,gates(row,6)));
                    circuit.push_back(qclab.qgates.RotationY(q,gates(row,4)));
                    circuit.push_back(qclab.qgates.Phase(q,gates(row,5)));
                else
                    assert(gates(row,1) == 2, 'Unknown gate encoding.');
                    circuit.push_back(qclab.qgates.CNOT(q,n-double(gates(row,3))));
                end
            end
            U = circuit.matrix(); V = U(:,1:half);
            expected = f.recovery_isometries{site};
            overlap = sum(conj(expected(:)).*V(:));
            assert(norm(V-(overlap/abs(overlap))*expected,'fro') < 1e-8);
            % Turn the simulated recovery columns into the complete spin map.
            p = site_permutation(n,site);
            compiled = zeros(d,d,4);
            for environment = 0:1
                R = V(environment*d+(1:d),:);
                for old_spin = 0:1
                    ordered = zeros(d);
                    ordered(:,old_spin*half+(1:half)) = R;
                    compiled(p,p,2*environment+old_spin+1) = ordered;
                end
            end
            operators{site} = compiled; circuits{site} = circuit;
        end
    end
    rho = f.rho_initial;
    for sweep = 1:double(f.sweeps)
        for site = 1:n
            K = operators{site}; next = zeros(d);
            for k = 1:size(K,3)
                next = next + K(:,:,k)*rho*K(:,:,k)';
            end
            rho = next;
        end
    end
    assert(norm(rho-f.rho_final,'fro') < 1e-8);
    result = struct('rho',rho,'rho_target',target, ...
        'trace_distance',sum(svd(rho-target))/2,'recovery_circuits',{circuits});
    fprintf('Quantum heat-bath replay matches Python; trace distance %.3e.\n',result.trace_distance);
end

function p = site_permutation(n,site)
    % p(new basis index) is the corresponding old row-major basis index.
    order = [site,1:site-1,site+1:n];
    p = zeros(1,2^n);
    for index = 0:2^n-1
        old = 0;
        for position = 1:n
            old = old + bitget(index,n-position+1)*2^(n-order(position));
        end
        p(index+1) = old+1;
    end
end

function H = reconstruct_hamiltonian(image,kappa,sigma)
    % Independent nearest-neighbor XXZ construction; supports the eight-spin
    % fixture without changing the original six-pixel demo's size guard.
    [height,width] = size(image); n = numel(image); H = zeros(2^n);
    X = [0,1;1,0]; Y = [0,-1i;1i,0]; Z = diag([1,-1]);
    for row = 1:height
        for col = 1:width
            for direction = 1:2
                rr = row+(direction==1); cc = col+(direction==2);
                if rr>height || cc>width, continue; end
                a = (row-1)*width+col; b = (rr-1)*width+cc;
                delta = 2-kappa*abs(double(image(row,col))-double(image(rr,cc)))/sigma;
                H = H-(two_site(X,a,b,n)+two_site(Y,a,b,n)+delta*two_site(Z,a,b,n))/4;
            end
        end
    end
end

function operator = two_site(P,a,b,n)
    operator = 1;
    for site = 1:n
        if site==a || site==b, local = P; else, local = eye(2); end
        operator = kron(operator,local);
    end
end
