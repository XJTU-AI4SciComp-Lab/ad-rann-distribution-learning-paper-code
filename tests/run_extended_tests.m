clear; clc;

this_file = mfilename('fullpath');
test_dir = fileparts(this_file);
root = fileparts(test_dir);
example_dir = fullfile(root,'examples','poisson_2d');

addpath(genpath(fullfile(root,'src')));
addpath(example_dir);
addpath(test_dir);

fprintf('Running legacy Gaussian tests...\n');
test_fast_vs_generic();
test_fast_gradient();

fprintf('Running extended generic/activation tests...\n');
test_random_weights_nd_compatibility();
test_poisson_fast_activations();
test_data_driven_nd();

fprintf('All extended tests passed.\n');
