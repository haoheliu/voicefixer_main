function rand(){
    min=$1
    max=$(($2-$min+1))
    num=$(($RANDOM+1000000000))
    echo $(($num%$max+$min))
}

# A Unique name for trail
rnd=$(rand 0 10000)

export TRAIL_NAME="a"$rnd
export BETTER_EXCEPTIONS=1
echo $TRAIL_NAME
echo "export TRAIL_NAME="$TRAIL_NAME >> ~/.bashrc

cp hdfs.sh ~

# Switch branch
git switch -c $TRAIL_NAME

# Add VPN
export http_proxy=10.20.47.147:3128  https_proxy=10.20.47.147:3128 no_proxy=code.byted.org

git clone https://github.com/tmux-plugins/tpm ~/.tmux/plugins/tpm

cp conf/.tmux.conf ~/
cp conf/.tmux.conf.local ~/


tmux source-file ~/.tmux.conf

# apt-get
chmod 777 /tmp
sudo apt-get update
pip3 install progressbar
sudo apt-get -y install zip
sudo apt-get -y install octave

cp conf/.vimrc ~/

# Add dependencies
git clone https://github.com/haoheliu/WavAugment.git
cd WavAugment
sudo python3 setup.py develop
cd ..

git clone https://github.com/jfsantos/SRMRpy.git
cd SRMRpy
pip3 install .
cd ..

git clone https://github.com/vBaiCai/python-pesq.git
cd python-pesq
pip3 install .
cd ..

source ~/hdfs.sh -get ../speechmetrics.tar
tar -zxvf speechmetrics.tar
cd speechmetrics
pip3 install .
cd ..


wget https://www.rarlab.com/rar/rarlinux-x64-6.0.1b1.tar.gz
tar -xzpvf rarlinux-x64-6.0.1b1.tar.gz
cd rar
sudo make
cd ..

pip3 install oct2py Cython # install Cython before installing pesq
pip3 install pesq


# PIP Install
pip3 install inplace-abn
pip3 install pretty-errors
pip3 install gitpython

pip3 install nnmnkwii
pip3 install kornia
pip3 install pyworld
pip3 install julius
pip3 install scikit-image
git clone https://github.com/ludlows/python-pesq.git
cd python-pesq
pip install .





