function error = verify_ising_gibbs_fixture(filename)
%VERIFY_ISING_GIBBS_FIXTURE Compare MATLAB enumeration with the Python export.
    fixture=load(filename);
    actual=algorithm1_ising_gibbs(fixture.image,fixture.alpha,fixture.beta_paper);
    assert(isequal(actual.edges-1,fixture.edges_zero_based));
    assert(max(abs(actual.delta(:)-fixture.delta(:)))<1e-12);
    assert(max(abs(actual.energies(:)-fixture.energies(:)))<1e-12);
    assert(max(abs(actual.weights(:)-fixture.weights(:)))<1e-12);
    error=max(abs(actual.gamma(:)-fixture.gamma(:)));
    assert(error<1e-12);
    fprintf('Ising Gibbs fixture agrees; maximum correlation error %.3g.\n',error);
end
