cd ../datasets/se
mkdir timit
cd timit
if [ ! -d "timit" ]; then
  echo "Fetching timit 44.1kHz dataset"
  hdfs dfs -get hdfs://haruna/home/byte_speech_sv/user/liuhaohe/dataloaders/timit.zip
  unzip timit.zip
fi

cd ../../../

python3 src/dataloaders/datasetParser/timit.py

echo "Done"