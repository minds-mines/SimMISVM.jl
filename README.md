# SimMISVM.jl

In this repository is the code associated with the publication titled "A Multi-Instance Support Vector Machine with Incomplete Data for Clinical Outcome Prediction of COVID-19" presented at the 12th ACM Conference on Bioinformatics, Computational Biology, and Health Informatics (BCB '21) held virtually from August 1-4, 2021.

This code base is using the [Julia Language (v1.6.0)](https://julialang.org/) to make a reproducible scientific project named
> SimMISVM.jl 

To (locally) reproduce this project, do the following:

0. Download this code base. 
1. Open a Julia console and do:
   ```julia
   julia> using Pkg
   julia> Pkg.activate("path/to/code")
   julia> Pkg.instantiate()
   ```

This will install all necessary packages for you to be able to run the scripts and
everything should work out of the box.

2. Run the tests associated with the SimMISVM model:
   ```julia
   julia> include("test/simmisvm_test.jl")
   ```
These tests ensure that the updates derived in Algorithm 2 are correct. E.g. since variable update is derived with respect to a primal variable, and the minimization is quadratic with respect to that variable, the Lagrangian should be a minimum after that variable has been updated. **Please note: The frist time this code is run it may take some extra time.**

3. For an example on running the SimMISVM.jl model on the COVID-19 dataset run:
   ```julia
   julia> include("simmisvm_example.jl") # This is the main "entry-point"
   ```

4. The code for the updates in Algorithm 2 are located in `src/SimMISVM.jl`.

5. The hyperparameter settings for each method-dataset pair for the results reported in Table 1 are as follows:

Models implemented from: https://github.com/alan-turing-institute/MLJ.jl

| Model | hyperparameter settings |
| ------- | ----- |
| kNN | `K = 7` |
| LightGBM | `learning_rate = 0.27, num_leaves = 32, max_depth = 24` |
| XGBoost | `eta = 0.22, max_depth = 8, lambda = 0.73, alpha = 0.44` |
| SVM | `C = 5e4, kernel = linear` |

Our model:

| Model | hyperparameter settings |
| ------- | ----- |
| MISVM | `C = 1e-3, μ=1e-4, ρ=1.2` |
| SimMISVM | `C = 10, α=0.01, β=0.01, μ=1e-4, ρ=1.2` |

## Other Files and Descriptions

 - `Manifest.toml` and `Project.toml` specify the project's dependencies.
 - `data/raw_results.csv` are used to generate Figure 2.
 - `data/time_series_375_prerpocess_en.csv` is the raw COVID-19 patient data provided in https://www.nature.com/articles/s42256-020-0180-7.
 - `data_utils.jl` contains functions that are used to assist in the handling of the temporal COVID-19 data. **For integrating a new dataset please refer to this file.**

## Issues

If you have any trouble with this code please [open a GitHub issue](https://github.com/minds-mines/SimMISVM.jl/issues) above.

## Citing This Work

TODO
