# README

## Overview
This project focuses on the analysis and visualization of medulloblastoma data using Variational Autoencoders (VAE). The pipeline includes data preprocessing, classification, clustering, and visualization steps.

## Table of Contents
- [Installation](#installation)
- [Usage](#usage)
- [Reproducing Results](#reproducing-results)
- [Contributing](#contributing)
- [License](#license)

## Installation
To set up the project, you need to create a conda environment using the provided `environment.yml` file. 

*Note: using mamba instead of conda is recommended for faster environment creation.*

Follow these steps:

1. Clone the repository:
    ```bash
    git clone https://github.com/gprolcastelo/scratchmedulloblastoma.git
    cd scratchmedulloblastoma
    ```

2. Create the conda environment:
    ```bash
    conda env create -f environment.yml
    ```

3. Activate the environment:
    ```bash
    conda activate medulloblastoma
    ```

## Usage
After setting up the environment, you can start using the scripts provided in the repository. The main pipeline script is `pipeline.sh`, which runs the entire analysis pipeline.

After following the steps from the [Installation](#installation) sectionn, just execute `pipeline.sh`:

```bash
    bash pipeline.sh
```

Finally, the Jupyter Notebook `putting_results_together.ipynb` contains the code for ad-hoc processes that were not included in the `src` folder codes.

## Reproducing Results
To reproduce the results, follow these steps:

1. Ensure that the conda environment is activated:
    ```bash
    conda activate medulloblastoma
    ```

2. Run the pipeline script:
    ```bash
    bash pipeline.sh
    ```

This script will execute all the necessary steps, including data preprocessing, classification, clustering, and visualization, and save the results to the `data` and `reports` subdirectories.

> **High Performance Computing (HPC) is highly recommended to run the pipeline**, as it requires significant computational resources, especially when training the VAE model and running the SHapley Additive exPlanations (SHAP) algorithm. 

For example, running the `src/classification_shap.py` on the preprocessed data took about 18 hours on 112 cores.

## Contributing
Contributions generally not expected. However, if you have any suggestions or improvements, feel free to reach out in the issues section. 
On the other hand, this is an open-source, open-science project, and so you are encouraged to reproduce our results.
Feel free to clone or fork the repository.

## License
This project is licensed under the Apache 2.0 License. See the [LICENSE](LICENSE) file for more details.
