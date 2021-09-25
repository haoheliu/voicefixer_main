cd ../datasets/se

if [ ! -d "vocal_wav_44k" ]; then
  echo "Fetching vocal_wav_44k 44.1kHz dataset"
  hdfs dfs -get hdfs://haruna/home/byte_speech_sv/user/liuhaohe/dataloaders/vocal_wav_44k.tar
  tar -zxvf vocal_wav_44k.tar
fi

cd ../../

python3 src/dataloaders/datasetParser/vocal_wav_44k.py

echo "Done"