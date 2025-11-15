interact --account=<VT USERNAME> --mem 1G --partition=<CURRENT PARTITION, ie: normal_q, a100_normal_q, etc>
module load Miniforge3
conda create -p ~/.conda/envs/SLEAP python=3.12
source activate ~/.conda/envs/SLEAP

pip install "sleap[nn]" --extra-index-url https://download.pytorch.org/whl/cu128 --index-url https://pypi.org/simple
pip install "sleap-io"
sleap-label --help
Note: This pip install is different if using GPU/CPU
To test sleap was imported
python -c "import sleap; sleap.versions()"

To load the conda environment, you have to go through the interact route every session
Then module load Miniforge3
Then you can load/create the environment
