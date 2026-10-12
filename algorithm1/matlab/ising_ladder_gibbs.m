function gamma = ising_ladder_gibbs(model,seeds)
%ISING_LADDER_GIBBS Exact 4x4 transfer contraction; O(L) work per seed.
% Seeds are one-based row-major pixel indices; Gamma=<Zi Zj>/4.
    [rows,L]=size(model.image);assert(rows==2,'Expected a two-row image.');
    validateattributes(seeds,{'numeric'},{'integer','>=',1,'<=',model.n});
    z=[1,1;1,-1;-1,1;-1,-1];vertical=zeros(1,L);horizontal=zeros(2,L-1);
    for e=1:size(model.edges,1)
        p=model.edges(e,1);q=model.edges(e,2);r=floor((p-1)/L)+1;c=mod(p-1,L)+1;
        if q==p+L,vertical(c)=model.delta(e)/4;else,horizontal(r,c)=model.delta(e)/4;end
    end
    logv=model.b*vertical'*(z(:,1).*z(:,2))';
    logv=logv-max(logv,[],2);v=exp(logv);T=cell(1,L-1);
    for c=1:L-1
        a=model.b*(horizontal(1,c)*(z(:,1)*z(:,1)')+horizontal(2,c)*(z(:,2)*z(:,2)'));
        a=a+logv(c+1,:);T{c}=exp(a-max(a,[],'all'));
    end
    f=zeros(L,4);g=zeros(L,4);nf=zeros(1,L-1);ng=zeros(1,L-1);
    f(1,:)=v(1,:)/sum(v(1,:));g(L,:)=1;
    for c=1:L-1,a=f(c,:)*T{c};nf(c)=sum(a);f(c+1,:)=a/nf(c);end
    for c=L-1:-1:1,a=T{c}*g(c+1,:)';ng(c)=sum(a);g(c,:)=a'/ng(c);end
    gamma=zeros(numel(seeds),2*L);
    for j=1:numel(seeds)
        sr=floor((seeds(j)-1)/L)+1;sc=mod(seeds(j)-1,L)+1;values=zeros(2,L);
        tag=f(sc,:).*z(:,sr)';
        for c=sc:L
            if c>sc,tag=tag*T{c-1}/nf(c-1);end
            for r=1:2,values(r,c)=(tag.*z(:,r)')*g(c,:)'/(f(c,:)*g(c,:)')/4;end
        end
        tag=g(sc,:)'.*z(:,sr);
        for c=sc-1:-1:1
            tag=T{c}*tag/ng(c);
            for r=1:2,values(r,c)=(f(c,:).*z(:,r)')*tag/(f(c,:)*g(c,:)')/4;end
        end
        values(sr,sc)=.25;gamma(j,:)=reshape(values.',1,[]);
    end
end
