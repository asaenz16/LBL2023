function result = algorithm1_ising_petz_qclab(L,varargin)
%ALGORITHM1_ISING_PETZ_QCLAB Algorithm 1 with immediate-neighbor Petz updates.
% Example: r=algorithm1_ising_petz_qclab(4,'Sweeps',50,'Simulation','density');
% L is a multiple of 4. Image = two copies of [0 0 255 255 ...].
% Simulation modes:
%   density: exact QCLAB-derived quantum channels, N<=8, including coherences.
%   population: exact diagonal circuit reduction, N<=20.
%   trajectories: independent circuit-population trajectories, linear memory.
%   none: construct local QCLAB circuits and schedule, no state simulation.
% Temperature lives in the conditional rotations; bath ancillas reset to |0>.
    p=inputParser;
    addParameter(p,'Alpha',.001);addParameter(p,'Threshold',100);addParameter(p,'K',.8);
    addParameter(p,'B',[]);addParameter(p,'Sweeps',50);addParameter(p,'Simulation','density');
    addParameter(p,'Initial','plus_x');addParameter(p,'Shots',8192);addParameter(p,'RngSeed',1200);
    parse(p,varargin{:});o=p.Results;
    validateattributes(L,{'numeric'},{'integer','scalar','>=',4});assert(mod(L,4)==0);
    validateattributes(o.Sweeps,{'numeric'},{'integer','scalar','nonnegative'});
    validateattributes(o.Shots,{'numeric'},{'integer','scalar','>=',2});
    assert(ismember(o.Initial,{'plus_x','mixed','all_up'}));
    assert(ismember(o.Simulation,{'density','population','trajectories','none'}));
    row=255*mod(floor((0:L-1)/2),2);image=[row;row];n=2*L;seeds=[1,3];
    model=ising_petz_qclab_model(image,'Alpha',o.Alpha,'Threshold',o.Threshold,'K',o.K,'B',o.B);
    result=struct('model',model,'seeds_one_based',seeds,'simulation',o.Simulation,'initial',o.Initial, ...
        'sweeps',0:o.Sweeps,'gamma_gibbs',ising_ladder_gibbs(model,seeds));
    if strcmp(o.Simulation,'none'),return;end
    gamma=zeros(o.Sweeps+1,numel(seeds),n);se=zeros(size(gamma));
    previous=rng;cleanup=onCleanup(@()rng(previous)); %#ok<NASGU>
    rng(o.RngSeed,'twister');
    if strcmp(o.Simulation,'trajectories')
        bits=false(n,o.Shots);
        if ~strcmp(o.Initial,'all_up'),bits=rand(n,o.Shots)>.5;end
    else
        if strcmp(o.Simulation,'density'),assert(n<=8,'Use population or trajectories for larger strips.');
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
    result.gamma_linf=squeeze(max(abs(gamma-reshape(result.gamma_gibbs,1,numel(seeds),n)),[],3));
    result.same_segment_by_seed=gamma>0;
    if strcmp(o.Simulation,'density'),result.rho_final=rho;
    elseif strcmp(o.Simulation,'population'),result.population_final=probability;end
    if strcmp(o.Simulation,'trajectories')
        result.shots=o.Shots;
        result.simultaneous_radius=sqrt(log(2*n/.05)/(8*o.Shots));
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
