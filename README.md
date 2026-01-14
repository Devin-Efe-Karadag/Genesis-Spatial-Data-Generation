conda create -n sandbox python=3.9
git clone https://github.com/Genesis-Embodied-AI/Genesis.git
cd Genesis
pip install -e .

git submodule update --init --recursive
pipt install -e ".[render]"

# install xhost (optional)
wget https://www.x.org/releases/individual/app/xhost-1.0.8.tar.gz
tar -xf xhost-1.0.8.tar.gz
cd xhost-1.0.8

# choose your own local path (optional)
./configure --prefix=PATH
make
make install
export PATH="PATH/bin:$PATH"

xhost +SI:localuser:$USER
