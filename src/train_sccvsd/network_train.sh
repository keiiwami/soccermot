python3 network_train.py \
--train-file './train_data_50k.mat' \
--cuda-id 0 \
--lr 0.01 \
--num-epoch 10 \
--batch-size 64 \
--num-batch 128 \
--random-seed 0 \
--resume 'network.pth' \
--save-name 'network.pth'


#python network_train.py --train-file '../../data/train_data_10k.mat' --cuda-id 0 --lr 0.01 --num-epoch 10 --batch-size 64 --num-batch 128 --random-seed 0 --resume '' --save-name 'network.pth'