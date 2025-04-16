# Action Extractor

Code for paper **Imitation Learning with Precisely Labeled Human Demonstrations**

## Installation

### 1. Clone the repository
```bash
git clone https://github.com/yourusername/action_extractor.git
cd action_extractor
```

### 2. Create and activate conda environment
```bash
# Create conda environment
conda create -n action_extractor python=3.9
conda activate action_extractor

# Install PyTorch with CUDA 11.8
conda install pytorch torchvision pytorch-cuda=11.8 -c pytorch -c nvidia
```

### 3. Install system dependencies
```bash
sudo apt-get update
sudo apt-get install -y \
    ffmpeg \
    libavcodec-dev \
    libavformat-dev \
    libavdevice-dev \
    python3-dev \
    libglew-dev \
    libosmesa6-dev \
    patchelf \
    gcc \
    make
```

### 4. Install MuJoCo
```bash
# Install mujoco
conda install -c conda-forge mujoco

# Set environment variables
echo 'export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$HOME/.mujoco/mujoco210/bin' >> ~/.bashrc
source ~/.bashrc
```

### 5. Install Python dependencies and action_extractor
```bash
# Install dependencies
pip install -r requirements.txt

# Install action_extractor in development mode
pip install -e .
pip install "cython<3"
```

## Quick Start

### 1. Download demo data
```bash
# Create data directory
mkdir -p data/manipulation_demos/point_cloud_datasets

# Download sample dataset (replace with actual download command)
wget https://huggingface.co/datasets/yilongsong/action-extractor-demos/resolve/main/square_d0_sample.hdf5 \
-O data/manipulation_demos/point_cloud_datasets/square_d0_sample.hdf5
```

### 2. Run demo visualization
```bash
python action_extractor/point_cloud/robosuite/visualize_pseudo_actions_rollouts.py
```

In the video saved, the left two quadrants are the ground truth, while the right two quadrants are visualizations of the rollout using the estimated pose!

## Project Structure

```
action_extractor/
├── action_extractor/       # Main package
│   ├── utility_scripts/    # Utility scripts for data processing
│   └── point_cloud/       # Core pose estimation code
├── requirements.txt        # Project dependencies
└── README.md              # This file
```

## Common Issues

1. **CUDA version mismatch**: Make sure your NVIDIA drivers support CUDA 11.8
2. **MuJoCo errors**: Ensure MuJoCo is properly installed and environment variables are set
3. **Missing dependencies**: Run `pip install -r requirements.txt` again

## License

This project is licensed under the MIT License.

<!-- ## Citation

If you use this code in your research, please cite:
```bibtex
@article{your-paper,
    title={Imitation Learning with Precisely Labeled Human Demonstrations},
    author={Your Name},
    year={2024}
}
``` -->