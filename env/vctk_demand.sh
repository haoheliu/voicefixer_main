cd ../datasets/se

if [ ! -d "vd_noise" ]; then
  echo "Fetching vctk-demand 44.1kHz dataset"
  hdfs dfs -get hdfs://haruna/home/byte_speech_sv/user/liuhaohe/dataloaders/vd_noise.tar
#  hdfs dfs -get hdfs://haruna/home/byte_speech_sv/user/liuhaohe/dataloaders/clean_trainset_56spk_wav.zip
  tar -zxvf vd_noise.tar
#  unzip clean_trainset_56spk_wav.zip
fi

cd ../../

python3 src/dataloaders/datasetParser/vctk_demand.py

echo "Done"