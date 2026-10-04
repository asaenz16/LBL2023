function test_algorithm1(fixture_filename)
%TEST_ALGORITHM1 Run in MATLAB with QCLAB on the path. Optional Python fixture.
    model = algorithm1_model([0,1], 0.5, 'Sigma',1);
    delta = 1.5;
    H = diag([-delta,delta,delta,-delta])/4;
    H(2,3) = -0.5; H(3,2) = -0.5;
    assert(norm(model.H-H,'fro') < 1e-12);
    model = algorithm1_model([0,12;47,90], 2, 'SigmaDDOF',0);
    assert(isequal(model.edges,[1,2;1,3;2,4;3,4]));
    assert(isequal(model.distances,[12;47;78;43]));
    assert(abs(model.sigma-std([12,47,78,43],1)) < 1e-12);
    result = algorithm1_qclab([0,0;255,255], 2, 0, 1, 'Shots',32);
    assert(norm(result.rho-eye(16)/16,'fro') < 1e-10);
    assert(max(abs(result.gamma_exact-[0.25,0,0,0])) < 1e-10);
    result = algorithm1_qclab([0,0], 2, 200, 1, 'Sigma',1, 'Shots',32);
    assert(norm(result.rho-diag([0.5,0,0,0.5]),'fro') < 1e-10);
    if nargin > 0, verify_python_fixture(fixture_filename); end
    fprintf('Algorithm 1 MATLAB checks passed.\n');
end
