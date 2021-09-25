cd ../datasets/mss

if [ ! -d "musdb18hq" ]; then
  echo "Fetching musdb18hq 44.1kHz dataset"
  hdfs dfs -get hdfs://haruna/home/byte_speech_sv/user/liuhaohe/dataloaders/musdb18hq.tar
  tar -xzvf musdb18hq.tar
fi

cd ../../

python3 src/dataloaders/datasetParser/musdb.py

echo "Done"
