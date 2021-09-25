cd ../datasets/se

if [ ! -d "wav48" ]; then
  echo "Fetching vctk 44.1kHz dataset"
  hdfs dfs -get hdfs://haruna/home/byte_speech_sv/user/liuhaohe/dataloaders/vctk_44k_92.tar
  tar -zxvf vctk_44k_92.tar
  rm wav48/test/p376/p376_295_mic1.flac.wav
fi

cd ../../

python3 src/dataloaders/datasetParser/vctk.py

echo "Done"