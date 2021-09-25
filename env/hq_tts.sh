cd ../datasets/se

if [ ! -d "hq_tts" ]; then
  echo "Fetching hq_tts 44.1kHz dataset"
  hdfs dfs -get hdfs://haruna/home/byte_speech_sv/user/liuhaohe/dataloaders/hq_tts.tar
  tar -zxvf hq_tts.tar
fi

cd ../../

python3 src/dataloaders/datasetParser/hq_tts.py

echo "Done"