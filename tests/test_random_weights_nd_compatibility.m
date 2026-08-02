function test_random_weights_nd_compatibility()
%TEST_RANDOM_WEIGHTS_ND_COMPATIBILITY 2-D generic draw matches legacy draw.

    domain = [-1 1;-2 3];
    seed = 17;
    m = 37;

    b_old = build_random_weights(m,domain,seed);
    b_nd = build_random_weights_nd(m,domain,seed);

    assert(isequal(b_old.Z,b_nd.Z));
    assert(isequal(b_old.C,b_nd.C));

    fprintf('2-D random-weight compatibility: exact match.\n');
end
