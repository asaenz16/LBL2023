function result = algorithm1_ising_image_qclab(image,varargin)
%ALGORITHM1_ISING_IMAGE_QCLAB Local Petz Algorithm 1 on arbitrary gray images.
% r=algorithm1_ising_image_qclab(image,'Seeds',[background foreground], ...
%     'Simulation','trajectories','Shots',4096,'Sweeps',500);
% Seeds are ONE-BASED ROW-MAJOR pixel indices. No spin is pinned.
% The bath rotations are native QCLAB gates; reset is an exact bath trace.
% Supply GibbsReference for a large 2D grid if reference correlations are needed.
% Without it the driver computes a reference for two-row images or <=16 spins.
    p=inputParser;
    addParameter(p,'Seeds',1);addParameter(p,'GibbsReference',[]);
    addParameter(p,'Alpha',.001);addParameter(p,'Threshold',100);addParameter(p,'K',.8);
    addParameter(p,'B',[]);addParameter(p,'Sweeps',50);addParameter(p,'Simulation','density');
    addParameter(p,'Initial','plus_x');addParameter(p,'Shots',8192);addParameter(p,'RngSeed',1200);
    parse(p,varargin{:});o=p.Results;
    validateattributes(o.Sweeps,{'numeric'},{'integer','scalar','nonnegative'});
    validateattributes(o.Shots,{'numeric'},{'integer','scalar','>=',2});
    assert(ismember(o.Initial,{'plus_x','mixed','all_up'}));
    assert(ismember(o.Simulation,{'density','population','trajectories','none'}));
    assert(ismatrix(image));n=numel(image);seeds=reshape(o.Seeds,1,[]);
    validateattributes(seeds,{'numeric'},{'integer','nonempty','>=',1,'<=',n});
    model=ising_petz_qclab_model(image,'Alpha',o.Alpha,'Threshold',o.Threshold,'K',o.K,'B',o.B);
    reference=o.GibbsReference;
    if isempty(reference) && size(image,1)==2
        reference=ising_ladder_gibbs(model,seeds);
    elseif isempty(reference) && n<=16
        states=uint64((0:2^n-1)');z=zeros(2^n,n);
        for i=1:n,z(:,i)=1-2*double(bitget(states,n-i+1));end
        exponent=zeros(2^n,1);
        for e=1:size(model.edges,1)
            exponent=exponent+model.b*model.delta(e)/4*z(:,model.edges(e,1)).*z(:,model.edges(e,2));
        end
        weights=exp(exponent-max(exponent));weights=weights/sum(weights);
        reference=(z(:,seeds).*weights)'*z/4;
    end
    if ~isempty(reference),assert(isequal(size(reference),[numel(seeds),n]));end
    result=struct('model',model,'seeds_one_based',seeds,'simulation',o.Simulation,'initial',o.Initial, ...
        'sweeps',0:o.Sweeps,'gamma_gibbs',reference);
    if strcmp(o.Simulation,'none'),return;end
    gamma=zeros(o.Sweeps+1,numel(seeds),n);se=zeros(size(gamma));
    previous=rng;cleanup=onCleanup(@()rng(previous)); %#ok<NASGU>
    rng(o.RngSeed,'twister');
    if strcmp(o.Simulation,'trajectories')
        bits=false(n,o.Shots);
        if ~strcmp(o.Initial,'all_up'),bits=rand(n,o.Shots)>.5;end
    else
        if strcmp(o.Simulation,'density'),assert(n<=9,'Use population or trajectories above 9 spins.');
        else,assert(n<=20,'Use trajectories above 20 spins.');end
        dim=2^n;state=uint64((0:dim-1)');z=zeros(dim,n);
        for i=1:n,z(:,i)=1-2*double(bitget(state,n-i+1));end
        probability=ones(dim,1)/dim;
        if strcmp(o.Initial,'all_up'),probability=zeros(dim,1);probability(1)=1;end
        if strcmp(o.Simulation,'density')
            if strcmp(o.Initial,'plus_x'),rho=ones(dim)/dim;else,rho=diag(probability);end
            embedded=cell(n,4);
            for i=1:n
                for j=1:4,embedded{i,j}=embed_local(model.kernels{i}.kraus{j},model.sites{i},n);end
            end
        end
    end
    for sweep=0:o.Sweeps
        if strcmp(o.Simulation,'trajectories')
            spins=1-2*double(bits);g=(spins(seeds,:)*spins')/(4*o.Shots);
            gamma(sweep+1,:,:)=g;se(sweep+1,:,:)=sqrt(max(0,1/16-g.^2)/(o.Shots-1));
            if sweep==0
                g=zeros(numel(seeds),n);
                if strcmp(o.Initial,'all_up'),g(:)=.25;end
                for j=1:numel(seeds),g(j,seeds(j))=.25;end
                gamma(1,:,:)=g;se(1,:,:)=0;
            end
        else
            if strcmp(o.Simulation,'density'),probability=real(diag(rho));end
            for j=1:numel(seeds),gamma(sweep+1,j,:)=(probability.*z(:,seeds(j)))'*z/4;end
        end
        if sweep==o.Sweeps,break;end
        for i=model.order
            if strcmp(o.Simulation,'density')
                next=zeros(size(rho));
                for j=1:4,a=embedded{i,j};next=next+a*rho*a';end
                rho=next;
            elseif strcmp(o.Simulation,'population')
                lo=find(bitget(state,n-i+1)==0);hi=lo+2^(n-i);
                index=zeros(numel(lo),1);nb=model.neighbors{i};
                for j=1:numel(nb),index=index+double(bitget(state(lo),n-nb(j)+1))*2^(j-1);end
                p0=model.probability{i}(index+1);total=probability(lo)+probability(hi);
                probability(lo)=p0.*total;probability(hi)=(1-p0).*total;
            else
                index=zeros(1,o.Shots);nb=model.neighbors{i};
                for j=1:numel(nb),index=index+double(bits(nb(j),:))*2^(j-1);end
                p0=reshape(model.probability{i}(index+1),1,[]);
                bits(i,:)=rand(1,o.Shots)>p0;
            end
        end
    end
    result.gamma=gamma;result.gamma_se=se;
    if ~isempty(reference)
        result.gamma_linf=squeeze(max(abs(gamma-reshape(reference,1,numel(seeds),n)),[],3));
    end
    result.same_segment_by_seed=gamma>0;
    if strcmp(o.Simulation,'density'),result.rho_final=rho;
    elseif strcmp(o.Simulation,'population'),result.population_final=probability;end
    if strcmp(o.Simulation,'trajectories')
        result.shots=o.Shots;
        result.simultaneous_radius=sqrt(log(2*n*numel(seeds)/.05)/(8*o.Shots));
    end
end

function full = embed_local(local,sites,n)
    k=numel(sites);[r,c,value]=find(local);dim=2^n;rows=[];cols=[];values=[];
    for source=0:dim-1
        localSource=0;cleared=uint64(source);
        for j=1:k
            localSource=2*localSource+double(bitget(uint64(source),n-sites(j)+1));
            cleared=bitset(cleared,n-sites(j)+1,0);
        end
        matches=find(c==localSource+1);
        for entry=matches'
            target=cleared;
            for j=1:k,target=bitset(target,n-sites(j)+1,bitget(uint64(r(entry)-1),k-j+1));end
            rows(end+1)=double(target)+1;cols(end+1)=source+1;values(end+1)=value(entry); %#ok<AGROW>
        end
    end
    full=sparse(rows,cols,values,dim,dim);
end
