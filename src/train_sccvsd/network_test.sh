python3 network_test.py \
--edge-image-file './train_data_50k.mat' \
--model-name 'network.pth' \
--batch-size 64 \
--cuda-id 0 \
--save-file 'feature_camera_50k.mat'


#python network_test.py --edge-image-file '../../data/train_data_10k.mat' --model-name 'network.pth' --batch-size 64 --cuda-id 0 --save-file 'feature_camera_10k.mat'