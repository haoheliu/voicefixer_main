cd ../datasets/se
mkdir musdb_noise

cd musdb_noise
if [ ! -d "musdb_noise" ]; then
  echo "Fetching musdb_noise 44.1kHz dataset"
  hdfs dfs -get hdfs://haruna/home/byte_speech_sv/user/liuhaohe/dataloaders/musdb_noise.tar
  tar -xzvf musdb_noise.tar
fi
cd ..

cd ../../

python3 src/dataloaders/datasetParser/musdb_noise.py

echo "Done"
