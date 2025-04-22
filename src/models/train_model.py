import os
import sys
import warnings
import numpy as np

from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning.strategies import DDPStrategy

import hydra

warnings.filterwarnings("ignore", category=UserWarning)
np.set_printoptions(threshold=sys.maxsize)

from model_def import *

from src.data.MyDataloader import *

os.environ["TORCH_DISTRIBUTED_DEBUG"] = "INFO"


Hydra_path = '../conf/'
@hydra.main(config_path=Hydra_path, config_name="train_cfg.yaml")
def main(cfg):

# ====================== load config params from hydra ======================================
    pl_checkpoints_path = os.getcwd() + '/'
    
# ======================================== main section ==================================================
    

    dm = EnhancementDataModule(cfg)
    stopping_threshold = None

    model = Pl_UNet_enhance_reg(cfg)
   
    ckpt_monitor = cfg.ckpt_monitor
    mode = 'min'
    filename = 'epoch-{epoch:02d}-val-loss-{val_loss:.3f}'

    checkpoint_callback = ModelCheckpoint(
        monitor=ckpt_monitor,
        filename=filename,
        save_last = cfg.save_last,
        save_top_k = cfg.save_top_k,
        mode=mode,
        verbose=cfg.verbose
    )
    
    stop_callback = EarlyStopping(
        monitor=ckpt_monitor,
        patience=cfg.patience,
        stopping_threshold=stopping_threshold,
        check_finite=False,
        mode=mode
    )

    trainer = Trainer(  
                        accelerator = 'cuda',
                        fast_dev_run=False, 
                        check_val_every_n_epoch=cfg.check_val_every_n_epoch, 
                        default_root_dir= pl_checkpoints_path,                       
                        callbacks=[stop_callback, checkpoint_callback], 
                        strategy=DDPStrategy(find_unused_parameters=True),
                        num_sanity_val_steps = 0,
                        max_epochs=-1,
                        reload_dataloaders_every_n_epochs=1
                     )

    trainer.fit(model, dm)
    checkpoint_callback.best_model_path

if __name__ == "__main__":
    main()