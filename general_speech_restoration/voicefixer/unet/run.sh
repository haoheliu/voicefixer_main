pwd

export TRAIL_NAME="pc"

python3 train.py -m  unet \
                  -l l1 \
                 -t vctk vd_noise \
                 -v gsr \
                 -t_type vocals noise \
                 --aug_sources vocals \
                 --aug_effects low_pass clip reverb_rir \
                 -c 2 \
                 --lr 0.001 \
                 --source_sample_rate_low 1500 \
                 --source_sample_rate_high 44100 \
                 --gamma 0.90 \
                 --dl FixLengthAugRandomDataLoader \
                 --sample_rate 44100 \
                 -n voicefixer_unet \
                 --save_metric_monitor val_loss \
                 --early_stop_tolerance 64 \
                 --batchsize 40 \

rm temp_path.json