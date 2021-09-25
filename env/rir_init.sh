cd ../datasets/se

if [ ! -d "size_1_12_rt60_0.05_1.0_sr_44100_5_meter" ]; then
  echo "Fetching RIR dataset: size_1_12_rt60_0.05_1.0_sr_44100_5_meter"
  hdfs dfs -get hdfs://haruna/home/byte_speech_sv/user/liuhaohe/dataloaders/size_1_12_rt60_0.05_1.0_sr_44100_5_meter.tar
  tar -zxvf size_1_12_rt60_0.05_1.0_sr_44100_5_meter.tar
fi

cd ../../

echo "Done"