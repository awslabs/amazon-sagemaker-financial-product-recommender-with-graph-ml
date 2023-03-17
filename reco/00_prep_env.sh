
# Env for dataprep
source activate python3
pip install pyarrow
conda deactivate

# Env for DGLKE, needs python 3.7
conda env remove --name dglke
conda create -y --name dglke -c anaconda python=3.7
source activate dglke
pip install awscli boto3 jupyterlab
pip install -r requirements.txt 
conda deactivate
