# Survival Analysis-powered AI in Risk-aware Data-driven Multi-objective Optimization for Geological CO<sub>2</sub> Storage

This repository contains the code used in the submission for [Computers & Geosciences](https://www.sciencedirect.com/journal/computers-and-geosciences).

Examples of how to run the code can be seen in the files [main_surrogate](src/main_surrogate.py) and [main_optimization](src/main_optimization.py).

- The file [main_surrogate](src/main_surrogate.py) calls the [SurrogateTrainingManager](src/SurrogateTrainingManager.py) to train the Cox Proportional Hazards surrogate.
It also allows for the deliberate truncation of the dataset through the `truncate_data` function.
The dataset used in the paper can be found in the assets of the latest release of this repository.

- The file [main_optimization](src/main_optimization.py) calls the [OptimizationManager](src/OptimizationManager.py) to run the optimization.
Depending on whether a `trained_model_path` was given, it then calls either the [SurrogateProblem](src/ProblemManager/SurrogateProblem.py) or the [SimulationProblem](src/ProblemManager/SimulationProblem.py).
It also plots results through `plot_results`, and allows for the simulation of the final Pareto front (in the case of surrogate-guided optimization) through `simulate_pareto_front`.

The packages necessary for our code can be found in the [requirements](requirements.txt) file.
We fix the versions that we used to run the experiments; one may remove the specific versions if so desired.
The data used to train the surrogates can be found in the assets of the latest [release](https://github.com/agurwicz/sa-surrogate-multiobjective-optimization/releases) of this repository.

We have run the experiments for the paper on an Apple MacBook Pro, with the Apple M3 Max and 96 GB of memory.
To run the commercial reservoir simulations, we developed [github.com/agurwicz/cmg-macos](https://github.com/agurwicz/cmg-macos).
