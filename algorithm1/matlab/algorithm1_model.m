function model = algorithm1_model(image, kappa, varargin)
%ALGORITHM1_MODEL Paper XXZ Hamiltonian in row-major, most-significant-first order.
% Sigma is std of UNIQUE nearest-neighbor distances; ddof=1 by default.
% Use 'Sigma',positive_value if edge-distance variance is zero/undefined.
    p = inputParser;
    addParameter(p, 'Sigma', [], @(x) isempty(x) || (isscalar(x) && isfinite(x) && x > 0));
    addParameter(p, 'SigmaDDOF', 1, @(x) isscalar(x) && ismember(x, [0, 1]));
    parse(p, varargin{:});
    validateattributes(image, {'numeric'}, {'real', 'finite', 'nonempty'});
    validateattributes(kappa, {'numeric'}, {'real', 'finite', 'scalar', 'nonnegative'});
    assert(ismatrix(image) || (ndims(image) == 3 && size(image,3) == 3), ...
        'Use a grayscale matrix or RGB array.');
    image = double(image);
    [rows, cols, channels] = size(image);
    n = rows * cols;
    assert(n <= 6, 'Dense reference is restricted to at most 6 pixels.');
    % permute before reshape: MATLAB otherwise uses column-major pixel order.
    colors = reshape(permute(image, [2,1,3]), n, channels);
    edges = zeros(0,2);
    for row = 1:rows
        for col = 1:cols
            site = (row-1)*cols + col;
            if col < cols, edges(end+1,:) = [site, site+1]; end %#ok<AGROW>
            if row < rows, edges(end+1,:) = [site, site+cols]; end %#ok<AGROW>
        end
    end
    distances = sqrt(sum((colors(edges(:,1),:) - colors(edges(:,2),:)).^2, 2));
    sigma = p.Results.Sigma;
    if isempty(sigma)
        assert(numel(distances) > p.Results.SigmaDDOF, ...
            'Too few edge distances: supply a positive Sigma explicitly.');
        sigma = std(distances, 1-p.Results.SigmaDDOF);
        assert(sigma > 0, 'Zero distance variance: supply a positive Sigma explicitly.');
    end
    delta = 2 - kappa * distances / sigma;
    X = [0,1;1,0]; Y = [0,-1i;1i,0]; Z = diag([1,-1]);
    paulis = {X,Y,Z};
    H = zeros(2^n);
    for e = 1:size(edges,1)
        for axis = 1:3
            term = 1;
            for site = 1:n
                if any(edges(e,:) == site), local = paulis{axis}; else, local = eye(2); end
                term = kron(term, local);
            end
            coefficient = 1;
            if axis == 3, coefficient = delta(e); end
            H = H - coefficient * term / 4;
        end
    end
    model = struct('image',image, 'n',n, 'kappa',kappa, 'sigma',sigma, ...
        'sigma_ddof',p.Results.SigmaDDOF, 'edges',edges, 'distances',distances, ...
        'delta',delta, 'H',H);
end
