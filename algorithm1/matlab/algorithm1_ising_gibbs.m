function result = algorithm1_ising_gibbs(image, alpha, beta)
%ALGORITHM1_ISING_GIBBS Exact diagonal-Ising Gibbs correlations for <=20 pixels.
% H=-sum_unique_edges atan(alpha*(distance-125))*Zp*Zq/4.
% beta is the PAPER parameter; the exponent uses beta/(rows*cols).
% Raw 0..255 grayscale or RGB channels, Euclidean distance, row-major pixels.
% Positive alpha literally implements the proposed sign; negative alpha
% favors like spins for similar colors with the paper's minus sign in H.
    validateattributes(image, {'numeric'}, {'real','finite','nonempty','>=',0,'<=',255});
    validateattributes(alpha, {'numeric'}, {'real','finite','scalar'});
    validateattributes(beta, {'numeric'}, {'real','finite','scalar','nonnegative'});
    assert(ismatrix(image) || (ndims(image)==3 && size(image,3)==3));
    image=double(image); [rows,cols,channels]=size(image); n=rows*cols;
    assert(n<=20, 'Exact enumeration is restricted to at most 20 pixels.');
    colors=reshape(permute(image,[2,1,3]),n,channels); edges=zeros(0,2);
    for r=1:rows
        for c=1:cols
            p=(r-1)*cols+c;
            if c<cols, edges(end+1,:)=[p,p+1]; end %#ok<AGROW>
            if r<rows, edges(end+1,:)=[p,p+cols]; end %#ok<AGROW>
        end
    end
    distances=sqrt(sum((colors(edges(:,1),:)-colors(edges(:,2),:)).^2,2));
    delta=atan(alpha*(distances-125));
    z=zeros(2^n,n); states=uint64((0:2^n-1)');
    for p=1:n, z(:,p)=1-2*double(bitget(states,n-p+1)); end
    energies=zeros(2^n,1); connected=logical(eye(n));
    for e=1:size(edges,1)
        p=edges(e,1); q=edges(e,2);
        energies=energies-delta(e)*z(:,p).*z(:,q)/4;
        if delta(e)~=0, connected(p,q)=true; connected(q,p)=true; end
    end
    for p=1:n, connected=connected | (connected(:,p) & connected(p,:)); end
    weights=exp(-(beta/n)*(energies-min(energies))); weights=weights/sum(weights);
    gamma=(z'*(weights.*z))/4; gamma=(gamma+gamma')/2;
    gamma(~connected)=0;
    if beta==0, gamma(:)=0; end
    gamma(1:n+1:end)=1/4;
    result=struct('image',image,'alpha',alpha,'beta_paper',beta, ...
        'beta_effective',beta/n,'edges',edges,'distances',distances,'delta',delta, ...
        'energies',energies,'weights',weights,'gamma',gamma,'magnetization',zeros(n,1), ...
        'same_segment_by_seed',gamma>0);
end
