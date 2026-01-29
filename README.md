# git clone https://github.com/daniel03c1/sandbox

# conda create -n sandbox python=3.10
# conda activate sandbox

conda install nvidia::cuda-toolkit==12.8.0
pip install torch==2.8.0 torchvision==0.23.0 --index-url https://download.pytorch.org/whl/cu128
pip install kaolin==0.18.0 -f https://nvidia-kaolin.s3.us-east-2.amazonaws.com/torch-2.8.0_cu128.html
pip install bpy==3.6.0 --extra-index-url https://download.blender.org/pypi/
pip install taichi


pip install git+https://github.com/Genesis-Embodied-AI/Genesis.git@v0.3.13
