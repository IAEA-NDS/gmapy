user_remote_awareness=$1
example_name=$2

if [ "$user_remote_awareness" != "remote" ]; then
    echo "This script is intended to be run remotely on an AWS machine." >&2
    echo "Add 'remote' as first command line argument if you know " \
         "what you are doing." >&2
    exit 1
fi

# disable interactive popups
# see https://stackoverflow.com/questions/73397110/how-to-stop-ubuntu-pop-up-daemons-using-outdated-libraries-when-using-apt-to-i 
sudo sed -i "/#\$nrconf{restart} = 'i';/s/.*/\$nrconf{restart} = 'a';/" \
         /etc/needrestart/needrestart.conf
sudo apt update
sudo apt install -y build-essential

# install the miniconda Python package manager
cd /home/ubuntu
curl -O https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh 
INSTALL_PATH=/home/ubuntu/miniconda
MINICONDA_SCRIPT=Miniconda3-latest-Linux-x86_64.sh
bash "$MINICONDA_SCRIPT" -b -p $INSTALL_PATH
source "$INSTALL_PATH/etc/profile.d/conda.sh"
conda init bash
source /home/ubuntu/.bashrc

# prepare the gmapy package (including the examples)
git clone -b usu_studies https://github.com/iaea-nds/gmapy
conda env create -f gmapy/environment.yml -n gmapy-conda
cd gmapy/examples/tensorflow/$example_name

# run calculation pipeline
echo "---- Starting the gmapy calculation pipeline ----"
screen -S gmapy_session -d -m
screen -S gmapy_session -X stuff "conda activate gmapy-conda$(printf \\r)"
screen -S gmapy_session -X stuff "python 01_model_preparation.py; touch finished_preparation$(printf \\r)"
screen -S gmapy_session -X stuff "python 02_parameter_optimization.py; touch finished_optimization$(printf \\r)"
screen -S gmapy_session -X stuff "python 03_mcmc_sampling.py; touch finished_sampling$(printf \\r)"
screen -S gmapy_session -X detach
