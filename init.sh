virtualenv env_voicefixer # create new environment
source env_voicefixer/bin/activate # activate environment

pip3 install -r requirements.txt # install requirements
pip3 install torch==1.8.1 torchaudio==0.8.1 pytorch_lightning==1.5.0
pip3 install git+https://github.com/aliutkus/speechmetrics#egg=speechmetrics[cpu]
pip3 install GitPython
sudo apt-get install libsox-fmt-all libsox-dev sox
python3 -m pip install git+https://github.com/facebookresearch/WavAugment.git

if ! [ -x "$(command -v unzip)" ]; then
  echo 'Error: unzip is not installed.' >&2
  exit 1
fi

if [ ! -d "datasets/se/wav48" ]; then
  echo "Preparing 44.1kHz speech train set"
  if [ ! -f "datasets/se/vctk.tar" ]; then
    wget https://zenodo.org/record/5528132/files/vctk.tar?download=1 -O datasets/se/vctk.tar
  fi
  tar -xzf datasets/se/vctk.tar -C datasets/se
fi
python3 datasets/datasetParser/vctk.py

if [ ! -d "datasets/se/vd_noise" ]; then
  echo "Preparing 44.1kHz noise train set"
  if [ ! -f "datasets/se/vd_noise.tar" ]; then
    wget https://zenodo.org/record/5528132/files/vd_noise.tar?download=1 -O datasets/se/vd_noise.tar
  fi
  tar -xzf datasets/se/vd_noise.tar -C datasets/se
fi
python3 datasets/datasetParser/vctk_demand.py

if [ ! -d "datasets/se/TestSets" ]; then
  echo "Preparing GSR and SSR test sets"
  if [ ! -f "datasets/se/GSR_and_SSR_testsets.zip" ]; then
    wget https://zenodo.org/record/5528144/files/GSR_and_SSR_testsets.zip?download=1 -O datasets/se/GSR_and_SSR_testsets.zip
  fi
  unzip -q datasets/se/GSR_and_SSR_testsets.zip -d datasets/se
fi
python3 datasets/datasetParser/test_set_speech_all_distortion.py

if [ ! -d "datasets/se/RIR_44k" ]; then
  echo "Preparing 44.1k Room Impulse Response dataset"
  if [ ! -f "datasets/se/RIR_44k.zip" ]; then
    wget https://zenodo.org/record/5528124/files/RIR_44k.zip?download=1 -O datasets/se/RIR_44k.zip
  fi
  unzip -q datasets/se/RIR_44k.zip -d datasets/se
fi


