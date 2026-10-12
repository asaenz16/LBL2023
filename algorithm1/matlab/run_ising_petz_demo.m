function result = run_ising_petz_demo()
%RUN_ISING_PETZ_DEMO Run and plot a native MATLAB/QCLAB 2x4 bath circuit.
% Run from any directory after adding algorithm1/matlab to the path.
    setup_ising_petz();here=fileparts(mfilename('fullpath'));
    output=fullfile(fileparts(here),'results','ising-petz');if ~isfolder(output),mkdir(output);end
    result=algorithm1_ising_petz_qclab(4,'Sweeps',50,'Simulation','density');
    fig=figure('Visible','off','Color','w','Position',[100,100,1150,760]);
    if isprop(fig,'Theme'),fig.Theme='light';end
    axes('Parent',fig,'Position',[.08,.58,.36,.26]);
    imagesc(result.model.image,[0,255]);axis image;colormap(gray);title('Input: alternating 2-by-2 blocks');
    axes('Parent',fig,'Position',[.59,.58,.36,.26]);
    imagesc(reshape(squeeze(result.gamma(end,1,:))<=0,4,2)');axis image;
    title('Segmentation after 50 sweeps');
    axes('Parent',fig,'Position',[.08,.1,.36,.31]);
    plot(result.sweeps,-squeeze(result.gamma(:,1,end)),'LineWidth',1.5);hold on;
    yline(-result.gamma_gibbs(1,end),'--','Exact Gibbs');xlabel('Sweeps');ylabel('-Gamma(seed,last)');grid on;
    axes('Parent',fig,'Position',[.59,.1,.36,.31]);
    semilogy(result.sweeps,max(result.gamma_linf,[],2),'LineWidth',1.5);
    xlabel('Sweeps');ylabel('Correlation error to Gibbs (infinity norm)');grid on;
    annotation(fig,'textbox',[.06,.94,.9,.04], ...
        'String','QCLAB Ising Petz circuit: 8 spins, exact density simulation', ...
        'EdgeColor','none','Color','k','FontSize',16,'HorizontalAlignment','center');
    annotation(fig,'textbox',[.06,.90,.9,.03], ...
        'String','Threshold 100; alpha 0.001; K = 0.8; all spins initially along +x', ...
        'EdgeColor','none','Color','k','FontSize',12,'HorizontalAlignment','center');
    for ax=findall(fig,'Type','axes')'
        set(ax,'Color','w','XColor','k','YColor','k','FontSize',11);
        ax.Title.Color='k';ax.XLabel.Color='k';ax.YLabel.Color='k';
    end
    exportgraphics(fig,fullfile(output,'matlab-demo.png'),'Resolution',180,'BackgroundColor','white');close(fig);
    gamma=result.gamma;gamma_gibbs=result.gamma_gibbs;rho_final=result.rho_final; %#ok<NASGU>
    save(fullfile(output,'matlab-demo.mat'),'gamma','gamma_gibbs','rho_final');
    fprintf('Saved native QCLAB demonstration to %s\n',output);
    fprintf('8 spins + 8 local bath qubits; 80 CNOT, 32 Ry, 24 reset operations per sweep.\n');
end
