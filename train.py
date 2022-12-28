import hydra
from omegaconf import OmegaConf
import wandb

@hydra.main(config_path='configs', config_name='train')
def main(cfg):
    print(OmegaConf.to_yaml(cfg))


if __name__ == "__main__":
    main()
