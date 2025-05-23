if [ ! -d "./logs" ]; then
    mkdir ./logs
fi

model_name=CDFM

root_path_name=./dataset/
data_path_name=ETTm2.csv
model_id_name=ETTm2
data_name=ETTm2
alpha=0.7

seq_len=512
pred_len=96
python -u run_longExp.py \
    --is_training 1 \
    --root_path $root_path_name \
    --data_path $data_path_name \
    --model_id  $model_name'_'$seq_len'_'$pred_len \
    --model $model_name \
    --data $data_name \
    --features M \
    --seq_len $seq_len \
    --pred_len $pred_len \
    --enc_in 7 \
    --des 'Exp' \
    --train_epochs 20\
    --patience 10\
    --alpha $alpha\
    --pre_epochs 2\
    --itr 1 --batch_size 128 --learning_rate 0.005 >logs/$model_id_name'_'$seq_len'_'$pred_len.log


seq_len=512
pred_len=192
python -u run_longExp.py \
    --is_training 1 \
    --root_path $root_path_name \
    --data_path $data_path_name \
    --model_id  $model_name'_'$seq_len'_'$pred_len \
    --model $model_name \
    --data $data_name \
    --features M \
    --seq_len $seq_len \
    --pred_len $pred_len \
    --enc_in 7 \
    --des 'Exp' \
    --train_epochs 20\
    --patience 10\
    --alpha $alpha\
    --pre_epoch 1\
    --itr 1 --batch_size 64 --learning_rate 0.001 >logs/$model_id_name'_'$seq_len'_'$pred_len.log


seq_len=336
pred_len=336
python -u run_longExp.py \
    --is_training 1 \
    --root_path $root_path_name \
    --data_path $data_path_name \
    --model_id  $model_name'_'$seq_len'_'$pred_len \
    --model $model_name \
    --data $data_name \
    --features M \
    --seq_len $seq_len \
    --pred_len $pred_len \
    --enc_in 7 \
    --des 'Exp' \
    --train_epochs 20\
    --patience 10\
    --alpha $alpha\
    --itr 1 --batch_size 128 --learning_rate 0.001 >logs/$model_id_name'_'$seq_len'_'$pred_len.log

seq_len=336
pred_len=720
python -u run_longExp.py \
    --is_training 1 \
    --root_path $root_path_name \
    --data_path $data_path_name \
    --model_id  $model_name'_'$seq_len'_'$pred_len \
    --model $model_name \
    --data $data_name \
    --features M \
    --seq_len $seq_len \
    --pred_len $pred_len \
    --enc_in 7 \
    --des 'Exp' \
    --train_epochs 20\
    --patience 10\
    --alpha $alpha\
    --itr 1 --batch_size 128 --learning_rate 0.001 >logs/$model_id_name'_'$seq_len'_'$pred_len.log