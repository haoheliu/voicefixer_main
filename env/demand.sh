cd ../datasets/se

if [ ! -d "demand" ]; then
  echo "Fetching demand 44.1kHz dataset"
  hdfs dfs -get hdfs://haruna/home/byte_speech_sv/user/liuhaohe/dataloaders/demand
  find . -iname "*.zip" -exec unzip {} \;
  find . -iname "*.zip" -exec rm {} \;
fi

cd ../../

python3 src/dataloaders/datasetParser/dcase.py

echo "Done"
