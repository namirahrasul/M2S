#!/bin/bash
BASELINE_RESAMPLE_FLAGS="--t_T 250 --n_sample 1 --jump_length 10  --jump_n_sample 10 --jump_interval 10"
MODEL_FLAGS="--num_heads 1 --attention_resolutions 16 --diffusion_steps 1000 --dropout 0.0 --image_size 256 --learn_sigma True --noise_schedule linear --num_channels 128 --num_head_channels 64 --num_res_blocks 1 --resblock_updown True --use_fp16 True --use_scale_shift_norm True"
SAMPLE_FLAGS="--timestep_respacing 270 --ddim_stride 5 --range_t 20 --batch_size 5"
BASIC_CONFIG_FLAGS="--use_inverse_masks False --use_ddim True --special_mask False" 
INPUT_PATH="--base_samples ./demo_split/celebahq/val --mask_path ./demo_split/mask/thick"
MODEL_PATH="--model_path_256 ../P2-weighting/checkpoint_256_celebahq/ema_0.9999_1000000.pt"
OUTPUT_PATH="--save_dir results_celebahq_256_thick_ours/celeba/thick"

CUDA_VISIBLE_DEVICES=1 python sample_256.py $BASELINE_RESAMPLE_FLAGS $CLASSIFIER_FLAGS $MODEL_FLAGS $SAMPLE_FLAGS $BASIC_CONFIG_FLAGS $INPUT_PATH $MODEL_PATH $CLASSIFIER_PATH $OUTPUT_PATH
