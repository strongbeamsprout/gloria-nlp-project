from omegaconf import OmegaConf
import gloria
from tqdm import tqdm


if __name__ == '__main__':
    cfg = OmegaConf.load('configs/imagenome_pretrain_config.yaml')
    dm = gloria.builder.build_data_module(cfg)
#    dl = dm.train_dataloader()
#    for batch in tqdm(dl, total=len(dl)):
#        pass
#    dl = dm.val_dataloader()
#    for batch in tqdm(dl, total=len(dl)):
#        pass
    d = dm.dm.val
    for i in tqdm(range(len(d)), total=len(d)):
        try:
            d[i]
        except Exception as e:
            print(e)
    print('done')
