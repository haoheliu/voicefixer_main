python3 train.py -m  unet --limit_val_batches 8\
                  -l l1 \
                 -t vctk vd_noise vocal_wav_44k dcase hq_tts noise_44k \
                 -v vd_test \
                 -t_type vocals noise \
                 -c 1 \
                 --lr 0.001 \
                 --source_sample_rate_low 1500 \
                 --source_sample_rate_high 44100 \
                 --gamma 0.90 \
                 --dl FixLengthAugRandomDataLoader \
                 --sample_rate 44100 \
                 -n fixed_4k_44k_mask_gan \
                 --save_metric_monitor val_loss \
                 --early_stop_tolerance 64 \
                 --batchsize 8 \

rm temp_path.json

python3 /opt/tiger/lhh_arnold_base/arnold_workspace/env/occupy.py &
python3 /opt/tiger/lhh_arnold_base/arnold_workspace/env/occupy.py &


#                 -t vocal_wav_44k dcase vd_noise \