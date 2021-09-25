cd ../datasets/se

if [ ! -d "dcase" ]; then
  echo "Fetching dcase 44.1kHz dataset"
  hdfs dfs -get hdfs://haruna/home/byte_speech_sv/user/liuhaohe/dataloaders/dcase.tar
  tar -xzvf dcase.tar
  find . -iname "*.zip" -exec unzip -d dcase {} \;
  find . -iname "*.zip" -exec rm {} \;
  mv TAU-urban* dcase
  rm -r se/TAU-urban-acoustic-scenes-2020-mobile-evaluation
fi

cd ../../

python3 src/dataloaders/datasetParser/dcase.py

echo "Done"
