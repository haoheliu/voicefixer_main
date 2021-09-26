if ! [ -x "$(command -v unzip)" ]; then
  echo 'Error: unzip is not installed.' >&2
  exit 1
fi

if [ ! -d "datasets/se/wav48" ]; then
  echo "Preparing 44.1kHz speech train set"
  if [ ! -f "datasets/se/vctk.tar" ]; then
    wget https://zenodo.org/record/5528132/files/vctk.tar?download=1 -O datasets/se/vctk.tar
  fi
  tar -xzvf datasets/se/vctk.tar
fi
python3 datasets/datasetParser/vctk.py

if [ ! -d "datasets/se/vd_noise" ]; then
  echo "Preparing 44.1kHz noise train set"
  if [ ! -f "datasets/se/vd_noise.tar" ]; then
    wget https://zenodo.org/record/5528132/files/vd_noise.tar?download=1 -O datasets/se/vd_noise.tar
  fi
  tar -xzvf datasets/se/vd_noise.tar
fi
python3 datasets/datasetParser/vctk_demand.py

if [ ! -d "datasets/se/Testsets" ]; then
  echo "Preparing GSR and SSR test sets"
  if [ ! -f "datasets/se/GSR_and_SSR_testsets.zip" ]; then
    wget https://zenodo.org/record/5528144/files/GSR_and_SSR_testsets.zip?download=1 -O datasets/se/GSR_and_SSR_testsets.zip
  fi
  unzip -xzvf datasets/se/GSR_and_SSR_testsets.zip
fi
python3 datasets/datasetParser/test_set_speech_all_distortion.py

if [ ! -d "datasets/se/RIR_44k" ]; then
  echo "Preparing 44.1k Room Impulse Response dataset"
  if [ ! -f "datasets/se/RIR_44k.zip" ]; then
    wget https://zenodo.org/record/5528124/files/RIR_44k.zip?download=1 -O datasets/se/RIR_44k.zip
  fi
  unzip -xzvf datasets/se/RIR_44k.zip
fi


