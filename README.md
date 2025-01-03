# Protein Stability ML Project
### by Anoch Mohan


## Environment Setup
Create the environment by:\
`conda env create -f environment.yml`

Activate it by:\
`conda activate ml_project`

## Getting the data
Access the data by going to [ProteinNet](https://github.com/aqlaboratory/proteinnet/blob/master/docs/raw_data.md) 

Select any of the ProteinNet datasets. I used ProteinNet7 as it was smaller in size. 

## Running the model
0. Open `script.ipynb`
0. Click on `Select Kernel` and choose 'ml_project'.\
*Make sure you had completed the environment setup before doing this step.*\
Follow the steps [here](https://code.visualstudio.com/docs/datascience/jupyter-kernel-management) for a better idea.
0. You can run the cells now.\
*Make sure the right file path is provided to access ProteinNet data.*

## Results
To view the results, go to [FoldMason Webserver](https://search.foldseek.com/foldmason) and select `Upload Previous Result (JSON)`. Then select `result.json` file.

### References
ProteinNet: AlQuraishi, M. ProteinNet: a standardized data set for machine learning of protein structure. BMC Bioinformatics 20, 311 (2019). https://doi.org/10.1186/s12859-019-2932-0

FoldMason: Gilchrist, Cameron LM, Milot Mirdita, and Martin Steinegger. "Multiple Protein Structure Alignment at Scale with FoldMason." bioRxiv (2024): 2024-08.
