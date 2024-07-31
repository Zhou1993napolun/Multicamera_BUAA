python3 tools/train_net.py --config-file ./configs/Wild/mgn_R50-ibn.yml --eval-only \
MODEL.WEIGHTS ./logs/wildsplit7/mgn_R50-ibn/model_best.pth MODEL.DEVICE "cuda:1"
