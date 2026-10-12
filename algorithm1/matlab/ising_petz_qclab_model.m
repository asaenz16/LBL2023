function model = ising_petz_qclab_model(image, varargin)
%ISING_PETZ_QCLAB_MODEL Bounded-size QCLAB circuits for the Ising Petz bath.
% H=-sum atan(Alpha*(Threshold-distance))*Zi*Zj/4; Alpha>0.
% B is beta_paper/N. Default K = B*median(abs(Delta))/4 = 0.8.
% Pixels/sites use ONE-BASED row-major indices. Local QCLAB qubit zero is
% the center, followed by sorted neighbors; the last qubit is the bath.
% A complete update is reset(bath), erase, reset(bath), recover, reset(bath).
% QCLAB supplies the elementary unitary gates. The driver handles resets
% by tracing and replacing the bath, retaining ALL measurement outcomes.
    setup_ising_petz();
    p = inputParser;
    addParameter(p,'Alpha',.001); addParameter(p,'Threshold',100);
    addParameter(p,'K',.8); addParameter(p,'B',[]); parse(p,varargin{:});
    alpha=p.Results.Alpha; threshold=p.Results.Threshold; b=p.Results.B;
    validateattributes(image,{'numeric'},{'real','finite','nonempty','>=',0,'<=',255});
    assert(ismatrix(image),'This implementation currently accepts grayscale images.');
    validateattributes(alpha,{'numeric'},{'real','finite','scalar','positive'});
    validateattributes(threshold,{'numeric'},{'real','finite','scalar','nonnegative'});
    [rows,cols]=size(image); n=rows*cols; colors=reshape(double(image).',[],1);
    edges=zeros(0,2);
    for r=1:rows
        for c=1:cols
            i=(r-1)*cols+c;
            if c<cols, edges(end+1,:)=[i,i+1]; end %#ok<AGROW>
            if r<rows, edges(end+1,:)=[i,i+cols]; end %#ok<AGROW>
        end
    end
    distances=abs(colors(edges(:,1))-colors(edges(:,2)));
    delta=atan(alpha*(threshold-distances)); scale=median(abs(delta))/4;
    if isempty(b)
        validateattributes(p.Results.K,{'numeric'},{'real','finite','scalar','positive'});
        assert(~isempty(delta) && scale>1e-12,'Nonzero median coupling required for K normalization.');
        b=p.Results.K/scale;
    end
    validateattributes(b,{'numeric'},{'real','finite','scalar','nonnegative'});
    neighbors=cell(1,n); sites=cell(1,n); probability=cell(1,n); angles=cell(1,n); kernels=cell(1,n);
    cache=containers.Map('KeyType','char','ValueType','any');
    for i=1:n
        idx=find(any(edges==i,2)); nb=zeros(1,numel(idx)); js=delta(idx)/4;
        for j=1:numel(idx), nb(j)=sum(edges(idx(j),:))-i; end
        [nb,order]=sort(nb); js=js(order); degree=numel(nb);
        config=zeros(2^degree,degree);
        for j=1:degree, config(:,j)=bitget(uint32((0:2^degree-1)'),j); end
        field=(1-2*config)*js;
        theta=2*atan2(exp(-max(b*field,0)),exp(min(b*field,0)));
        probability{i}=cos(theta/2).^2; angles{i}=theta;
        neighbors{i}=nb; sites{i}=[i,nb];
        key=sprintf('%.17g,',theta);
        if ~isKey(cache,key), cache(key)=local_kernel(theta); end
        kernels{i}=cache(key);
        probability{i}=kernels{i}.p0; % computed from the actual QCLAB gate matrices
    end
    if rows==2, count=4; else, count=5; end
    layers=cell(1,count);
    for i=1:n
        r=floor((i-1)/cols);c=mod(i-1,cols);
        if rows==2, color=mod(c+2*r,4); else, color=mod(r+2*c,5); end
        layers{color+1}(end+1)=i;
    end
    layers=layers(~cellfun(@isempty,layers));
    for layer=layers
        used=[sites{layer{1}}]; assert(numel(unique(used))==numel(used),'Overlapping stars.');
    end
    counts=struct('cx',0,'ry',0,'reset',3*n,'system_qubits',n,'parallel_bath_qubits',n);
    for i=1:n
        counts.cx=counts.cx+kernels{i}.cx;
        counts.ry=counts.ry+kernels{i}.ry;
    end
    model=struct('image',double(image),'n',n,'alpha',alpha,'threshold',threshold, ...
        'b',b,'beta_paper',b*n,'k',b*scale,'edges',edges,'delta',delta, ...
        'neighbors',{neighbors},'sites',{sites},'probability',{probability}, ...
        'angles',{angles},'kernels',{kernels},'layers',{layers},'order',[layers{:}], ...
        'gate_counts_per_sweep',counts,'unique_local_circuits',cache.Count);
end

function kernel = local_kernel(theta)
    degree=round(log2(numel(theta))); n=degree+1; bath=n; center=0;
    erase=qclab.QCircuit(n+1);
    erase.push_back(qclab.qgates.CNOT(center,bath));
    erase.push_back(qclab.qgates.CNOT(bath,center));
    erase.push_back(qclab.qgates.CNOT(center,bath));
    recover=qclab.QCircuit(n+1); count=2^degree; nry=0; ncx=4;
    for j=0:count-1
        gray=bitxor(uint32(j),bitshift(uint32(j),-1)); signs=ones(count,1);
        for i=0:count-1
            parity=0;
            for q=1:degree, parity=parity+double(bitget(bitand(uint32(i),gray),q)); end
            signs(i+1)=(-1)^parity;
        end
        phi=signs'*theta/count;
        if abs(phi)>1e-14
            recover.push_back(qclab.qgates.RotationY(center,phi)); nry=nry+1;
        end
        if degree>0
            next=uint32(mod(j+1,count)); nextGray=bitxor(next,bitshift(next,-1));
            change=bitxor(gray,nextGray); control=round(log2(double(change)))+1;
            recover.push_back(qclab.qgates.CNOT(control,center));ncx=ncx+1;
        end
    end
    recover.push_back(qclab.qgates.CNOT(center,bath));
    % Explicit bath contractions of QCLAB's gate matrices. Bath is the LSB.
    E=full(erase.matrix()); R=full(recover.matrix()); dim=2^n; kraus=cell(1,4);
    for s=0:1
        for t=0:1
            kraus{2*s+t+1}=sparse(R(s+1:2:end,1:2:end)*E(t+1:2:end,1:2:end));
        end
    end
    complete=sparse(dim,dim);
    for j=1:4, complete=complete+kraus{j}'*kraus{j}; end
    assert(norm(complete-speye(dim),'fro')<1e-11,'Local circuit is not trace preserving.');
    rest=2^degree;reverse=zeros(rest,1);
    for i=0:rest-1
        for j=1:degree,reverse(i+1)=reverse(i+1)+double(bitget(uint32(i),degree-j+1))*2^(j-1);end
    end
    gateP0=zeros(1,dim);
    for j=1:4,gateP0=gateP0+full(sum(abs(kraus{j}(1:rest,:)).^2,1));end
    expected=repmat(cos(theta(reverse+1)/2).^2,2,1)';
    assert(max(abs(gateP0-expected))<1e-12,'Conditional probabilities differ from the gate circuit.');
    p0=reshape(gateP0(reverse+1),[],1);
    kernel=struct('erase',erase,'recover',recover,'kraus',{kraus},'cx',ncx,'ry',nry, ...
        'n_patch_spins',n,'n_qubits',n+1,'reset_bath_qubit',bath,'p0',p0);
end
