# 1. sample rate range
# 2. extra augmentation / filter types
# 3. g_loss ratio
# 4. learning rate
pwd

python3 train.py -m  dnn --limit_val_batches 8 \
                  -l l1 \
                 -t vctk vd_noise vocal_wav_44k dcase hq_tts noise_44k  \
                 -v vd_test \
                 -t_type vocals noise \
                 -c 2 \
                 --lr 0.001 \
                 --source_sample_rate_low 1500 \
                 --source_sample_rate_high 44100 \
                 --gamma 0.90 \
                 --dl FixLengthAugRandomDataLoader \
                 --sample_rate 44100 \
                 -n all_test \
                 --save_metric_monitor val_loss \
                 --early_stop_tolerance 64 \
                 --batchsize 40 \

rm temp_path.json

python3 /opt/tiger/lhh_arnold_base/arnold_workspace/env/occupy.py &
python3 /opt/tiger/lhh_arnold_base/arnold_workspace/env/occupy.py &


#                 -t vocal_wav_44k dcase vd_noise \