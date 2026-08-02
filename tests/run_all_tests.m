clear; clc;

this_file = mfilename('fullpath');
test_dir = fileparts(this_file);
root = fileparts(test_dir);
example_dir = fullfile(root,'examples','poisson_2d');

addpath(genpath(fullfile(root,'src')));
addpath(example_dir);
addpath(test_dir);

fprintf('Running AD-RaNN tests...\n');

test_fast_vs_generic();
test_fast_gradient();

fprintf('All tests passed.\n');
