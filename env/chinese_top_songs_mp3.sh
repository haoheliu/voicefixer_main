cd ../datasets/se

if [ ! -d "test_meta" ]; then
  echo "Fetching chinese music top song 44.1kHz dataset"
  hdfs dfs -get hdfs://haruna/home/byte_speech_sv/user/liuhaohe/dataloaders/chinese_top_songs_mp3.tar
  tar -xzvf chinese_top_songs_mp3.tar
fi

cd ../../

python3 src/dataloaders/datasetParser/chinese_top_songs.py

echo "Done"
