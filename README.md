## RESULT

SCCvSDをSoccerNetのデータに適応

![](./result/1666031017.gif)

![](./result/1666033456.gif)



# MEMO

## GANの学習
```
python3 train_two_pix2pix.py --dataroot ./datasets/soccer_seg_detection --name soccer_seg_detection_pix2pix --model two_pix2pix --which_model_netG unet_256 --which_direction AtoB --lambda_A 100 --dataset_mode two_aligned --no_lsgan --norm batch --pool_size 0 --output_nc 1 --phase1 train_phase_1 --phase2 train_phase_2 --save_epoch_freq 2 --gpu_ids -1
```
## GANのテスト
```
python3 test_two_pix2pix.py --dataroot ./datasets/soccer_seg_detection --which_direction AtoB --model two_pix2pix --name soccer_seg_detection_pix2pix --output_nc 1 --dataset_mode aligned --which_model_netG unet_256 --norm batch --how_many 186 --loadSize 256  --gpu_ids -1 --which_epoch 200
```

## SCCvSDのテスト
```
python3 demo.py --feature-type 'deep' --query-index 0
```

