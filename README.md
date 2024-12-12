# Quadrotor Diffusion

## Setup

```bash
# Create conda environemnt
conda create -n qdiff python=3.10
conda activate qdiff

# Install core modules
cd quadrotor_diffusion
pip install -e .

# Install prerequisites for simulator
conda install -c anaconda gmp

# Install Simulator
cd simulator
pip install -e .
cd ..
git clone https://github.com/utiasDSL/pycffirmware.git  # note this is gitignored

# Optional: Use Apptainer for building pycffirmware
#           if you use apptainer clone repository into home directory 
cd
apptainer build qdiff.sif quadrotor-diffusiion/apptainer/quadrotor_diffusion.def

####################  add to top of .bashrc file  ####################
if [ -n "$APPTAINER_CONTAINER" ]; then
        # >>> conda initialize >>> (replace this with your conda installation)
        __conda_setup="$('/sw/ubuntu-22.04/anaconda3/2023.09/bin/conda' 'shell.bash' 'hook' 2> /dev/null)"
        if [ $? -eq 0 ]; then
            eval "$__conda_setup"
        else
            if [ -f "/sw/ubuntu-22.04/anaconda3/2023.09/etc/profile.d/conda.sh" ]; then
                . "/sw/ubuntu-22.04/anaconda3/2023.09/etc/profile.d/conda.sh"
            else
                export PATH="/sw/ubuntu-22.04/anaconda3/2023.09/bin:$PATH"
            fi
        fi
        unset __conda_setup
        # <<< conda initialize <<<

        conda activate qdiff
        export PS1='Apptainer> [\w]$ '

        return
fi
#######################  rest of .bashrc file  #######################

apptainer instance start --nv qdiff.sif qdiff
apptainer shell instance://qdiff
conda activate qdiff

# If you are building pycffirmware directly install these packages
sudo apt update && sudo apt install -y swig build-essential

cd quadrotor-diffusion/pycffirmware
git submodule update --init --recursive
cd wrapper
sh build_linux.sh

# At this point, apptainer instance can be stopped
apptainer instance stop qdiff
```

## Acknowledgements

This simulator code is a fork of [safe_control_gym](https://google.com) from University of Toronto's Dynamic Systems Lab / Vector Institute for Artificial Intelligence. The diffusion model implementation is based on [diffuser](https://github.com/jannerm/diffuser).