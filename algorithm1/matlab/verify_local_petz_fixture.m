function result = verify_local_petz_fixture(filename)
%VERIFY_LOCAL_PETZ_FIXTURE Dense MATLAB replay of bounded-patch XXZ updates.
% The exported pixel indices are zero-based; dense matrices use row-major
% pixel tensor factors, matching MATLAB kron. Only small references fit here.
% MATLAB execution has not been performed in the development workspace.
    f = load(filename); n = double(f.n_pixels); d = 2^n;
    assert(n<=8, 'This replay is for at most eight system spins.');
    operators = cell(1,n);
    for center = 1:n
        sites = double(f.sites_zero_based{center});
        local = f.kraus{center}; operators{center} = zeros(d,d,size(local,3));
        complete = zeros(size(local,1));
        for j = 1:size(local,3)
            complete = complete + local(:,:,j)'*local(:,:,j);
            operators{center}(:,:,j) = embed(local(:,:,j),sites,n);
        end
        assert(norm(complete-eye(size(complete)),'fro')<1e-10);
    end
    rho = f.rho_initial;
    for sweep = 1:double(f.sweeps)
        for center = double(f.order_zero_based)+1
            next = zeros(d); K = operators{center};
            for j = 1:size(K,3), next = next+K(:,:,j)*rho*K(:,:,j)'; end
            rho = next;
        end
    end
    error = norm(rho-f.rho_final,'fro');
    assert(error<1e-9, 'Local channel replay differs from Python.');
    assert(abs(trace(rho)-1)<1e-9);
    result = struct('rho',rho,'replay_error',error, ...
        'trace_distance_to_gibbs',sum(svd(rho-f.rho_gibbs))/2);
    fprintf('Local Petz replay matches; distance to global Gibbs = %.6f.\n',result.trace_distance_to_gibbs);
end

function K = embed(local,sites,n)
    k = numel(sites); d = 2^n; K = zeros(d);
    for source = 0:d-1
        local_source = 0; cleared = source;
        for j = 1:k
            local_source = 2*local_source+bitget(source,n-sites(j));
            cleared = bitset(cleared,n-sites(j),0);
        end
        for local_target = 0:2^k-1
            target = cleared;
            for j = 1:k
                target = bitset(target,n-sites(j),bitget(local_target,k-j+1));
            end
            K(target+1,source+1) = local(local_target+1,local_source+1);
        end
    end
end
