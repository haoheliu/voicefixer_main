cd ../datasets/se

if [ ! -d "noisy_testset_wav" ]; then
  echo "Fetching vctk-demand_test 44.1kHz dataset"
  mkdir vd_test
  cd vd_test
  hdfs dfs -get hdfs://haruna/home/byte_speech_sv/user/liuhaohe/dataloaders/DS_10283_1942.zip
  unzip DS_10283_1942.zip
  unzip noisy_testset_wav.zip
  unzip clean_testset_wav.zip
  find . -iname "*.zip" -exec rm {} \;
  cd ..
fi

cd ../../

python3 src/dataloaders/datasetParser/vd_test.py

echo "Done"