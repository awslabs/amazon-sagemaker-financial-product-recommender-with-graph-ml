# Financial Product Recommender with Graph ML

Institutional investment managers are required by the US Securities and Exchange Commission (SEC) 
to report their security holdings quarterly in form 13F.
In this workshop we will use these public reports to deduct aggregate trades by investors and 
build a trade recommendation model with graph machine learning.

To run the notebooks you first need to prepare the environment, then you can go through the python notebooks.

We use data from https://www.sec.gov and https://dataverse.harvard.edu (see reference).


## Prepare the environment

Start a notebook instance and prepare the instance environment.

In your AWS account, go to Services -> Amazon SageMaker -> Notebook -> Notebook instances -> Create notebook instance 

You need to input a name and an instance type. Optionally, you can increase the volume size. 
```
Notebook instance name: <enter a name for your instance here>
Notebook instance type: ml.c5.4xlarge
Additional configuration -> Volume size in GB: 10
```

You can use the instances below. They differ in terms of CPU capacity but have the same memory capacity (32GB).

- ml.c5.4xlarge (compute optimized)
- ml.m5.2xlarge (standard instance)
- ml.r5.xlarge (memory optimized)


When the instance is up and running, click Open JupyterLab, then open a Launcher with '+'

In the Launcher, click on Terminal, and run the following commands:

```
cd ~/SageMaker 
git clone https://github.com/awslabs/amazon-sagemaker-financial-product-recommender-with-graph-ml.git graph-reco
cd graph-reco/reco   
source ./00_prep_env.sh
```

The script above does 2 things:

- Update the conda *python3* environment with parquet support
- Create a new conda environment *dglke* for graph ML


## Go through the notebooks

Once the environment is prepared, go to the file browser (left pane) and open the folder *reco*.

Now you can run the notebooks from 10 to 50:

```
10_get_data.ipynb
20_prepare_trades.ipynb
30_train_remotely.ipynb
40_train_locally.ipynb (optional)
50_cleanup.ipynb (optional)
```

NB:
- 10, 20, 30 must be run in sequence
- 40 is optional and can be run at the same time as 30
- 50 is optional, run it to restart from scratch or cleanup when you are done

## Notebook tips

A notebook is composed of cells that can be executed one by one. 

To open a notebook, double click on the notebook name in the left pane.

To run though a notebook cell by cell, type Shift-Enter repeatedly.

To go back to a particular cell, select the cell or use the up/down keys.

To run a full notebook in one command, select the menu item Run -> Run All

To restart a notebook, select the menu item Kernel -> Restart Kernel

## References

```
@data{DVN/ZRH3EU_2020,
author = {Backus, Matthew and Conlon, Christopher T and Sinkinson, Michael},
publisher = {Harvard Dataverse},
title = {{Common Ownership Data: Scraped SEC form 13F filings for 1999-2017}},
UNF = {UNF:6:YGhiiDQlgB5GRDaun4olzA==},
year = {2020},
version = {V1},
doi = {10.7910/DVN/ZRH3EU},
url = {https://doi.org/10.7910/DVN/ZRH3EU}
}
```

## Security

See [CONTRIBUTING](CONTRIBUTING.md#security-issue-notifications) for more information.

## License

This library is licensed under the MIT-0 License. See the LICENSE file.

