
# Update base environment

source activate python3
pip install pyarrow
pip install thefuzz[speedup]
conda deactivate

# Create environment for DGL-KE, needs python 3.7

conda env remove --name dglke
conda create -y --name dglke -c anaconda python=3.7
source activate dglke
pip install awscli boto3 jupyterlab
pip install -r requirements.txt 
conda deactivate
