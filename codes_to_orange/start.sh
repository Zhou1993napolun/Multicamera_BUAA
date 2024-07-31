# export CUDA_VISIBLE_DEVICES='2'

# CUDA_VISIBLE_DEVICES=2 
python3 tools/train_net.py --config-file ./configs/Wild/mgn_R50-ibn.yml MODEL.DEVICE 'cuda:1' #Market1501/bagtricks_R50-ibn.yml # 
