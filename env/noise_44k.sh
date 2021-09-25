cd ../datasets/se

if [ ! -d "noise" ]; then
  echo "Fetching noise 44.1kHz dataset"
  hdfs dfs -get hdfs://haruna/home/byte_speech_sv/user/liuhaohe/dataloaders/AS-N.tar
  tar -xzvf AS-N.tar
fi

cd ../../

python3 src/dataloaders/datasetParser/as_n.py

echo "Done"
