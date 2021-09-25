cd ../datasets/se

mkdir speech_all_test_set
cd speech_all_test_set

if [ ! -d "TestSets" ]; then
  echo "Fetching TestSets dataset"
  hdfs dfs -get hdfs://haruna/home/byte_speech_sv/user/liuhaohe/dataloaders/test_set_speech_all_distortion_v1.4.zip
  unzip test_set_speech_all_distortion_v1.4.zip
fi

cd ../../../

python3 src/dataloaders/datasetParser/test_set_speech_all_distortion.py


echo "Done"