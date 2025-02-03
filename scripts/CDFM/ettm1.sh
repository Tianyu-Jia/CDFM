if [ ! -d "./logs" ]; then
    mkdir ./logs
fi

seq_len=96
model_name=CDFM

root_path_name=./dataset/
data_path_name=ETTm1.csv
model_id_name=ETTm1
data_name=ETTm1
alpha=0.3

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
    --train_epochs 10\
    --patience 10\
    --alpha $alpha\
    --itr 1 --batch_size 32 --learning_rate 0.001 >logs/$model_id_name'_'$seq_len'_'$pred_len.log 


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
    --train_epochs 10\
    --patience 10\
    --alpha $alpha\
    --itr 1 --batch_size 32 --learning_rate 0.001 >logs/$model_id_name'_'$seq_len'_'$pred_len.log 



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
    --train_epochs 10\
    --patience 10\
    --alpha $alpha\
    --itr 1 --batch_size 32 --learning_rate 0.001 >logs/$model_id_name'_'$seq_len'_'$pred_len.log 


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
    --train_epochs 10\
    --patience 10\
    --alpha $alpha\
    --itr 1 --batch_size 32 --learning_rate 0.001 >logs/$model_id_name'_'$seq_len'_'$pred_len.log 