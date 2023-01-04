import warnings
warnings.filterwarnings('ignore')
import hydra, wandb, collections, os
from omegaconf import OmegaConf
wandb.init(project="MIDOG2022")
import numpy as np
import torch
import torch.optim as optim
from torchvision import transforms
from retinanet import model
from dataset.dataset import CocoDataset
from dataset.utils import collater, Resizer, AspectRatioBasedSampler, Augmenter, Normalizer
from torch.utils.data import DataLoader
from retinanet import coco_eval
from tqdm.auto import tqdm


def train(cfg):
    dataset_train = CocoDataset(cfg.dataset.root, set_name='training.json',
                                transform=transforms.Compose([Normalizer(), Augmenter(), Resizer()]))
    dataset_val = CocoDataset(cfg.dataset.root, set_name='training.json',
                              transform=transforms.Compose([Normalizer(), Resizer()]))

    sampler = AspectRatioBasedSampler(dataset_train, batch_size=24, drop_last=False)
    dataloader_train = DataLoader(dataset_train, num_workers=3, collate_fn=collater, batch_sampler=sampler)

    if dataset_val is not None:
        sampler_val = AspectRatioBasedSampler(dataset_val, batch_size=1, drop_last=False)
        dataloader_val = DataLoader(dataset_val, num_workers=3, collate_fn=collater, batch_sampler=sampler_val)

    # Create the model
    if cfg.model.classification.depth == 18:
        retinanet = model.resnet18(num_classes=cfg.model.classification.num_classes, pretrained=True)
    elif cfg.model.classification.depth == 34:
        retinanet = model.resnet34(num_classes=cfg.model.classification.num_classes, pretrained=True)
    elif cfg.model.classification.depth == 50:
        retinanet = model.resnet50(num_classes=cfg.model.classification.num_classes, pretrained=True)
    elif cfg.model.classification.depth == 101:
        retinanet = model.resnet101(num_classes=cfg.model.classification.num_classes, pretrained=True)
    elif cfg.model.classification.depth == 152:
        retinanet = model.resnet152(num_classes=cfg.model.classification.num_classes, pretrained=True)
    else:
        raise ValueError('Unsupported model depth, must be one of 18, 34, 50, 101, 152')

    if cfg.training.use_gpu and torch.cuda.is_available():
        retinanet = retinanet.cuda()

    retinanet.training = True
    optimizer = optim.Adam(retinanet.parameters(), lr=1e-5)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=3, verbose=True)
    loss_hist = collections.deque(maxlen=500)

    retinanet.train()
    retinanet.freeze_bn()
    for epoch_num in range(cfg.training.epochs):

        retinanet.train()
        retinanet.freeze_bn()

        epoch_loss = []
        epoch_classification_loss = list()
        epoch_regression_loss = list()
        total_epoch_loss = list()

        pbar = tqdm(dataloader_train, colour="green", leave=True, position=0)

        for iter_num, data in enumerate(pbar):
            try:
                optimizer.zero_grad()
                classification_loss, regression_loss = retinanet([data['img'].cuda().float(), data['annot'].cuda()])

                classification_loss = classification_loss.mean()
                regression_loss = regression_loss.mean()

                loss = classification_loss + regression_loss

                epoch_classification_loss.append(classification_loss)
                epoch_regression_loss.append(regression_loss)
                total_epoch_loss.append(loss)

                if bool(loss == 0):
                    continue

                loss.backward()

                torch.nn.utils.clip_grad_norm_(retinanet.parameters(), 0.1)

                optimizer.step()
                loss_hist.append(float(loss))
                epoch_loss.append(float(loss))

                wandb.log({
                    "Claasification Loss": classification_loss,
                    "Regression Loss": regression_loss,
                    "Running Loss": loss,
                })

                pbar.set_description(
                    'Epoch: {}/{} |  C_loss: {:1.2f} | R_loss: {:1.2f} | Running loss: {:1.2f}'.format(
                        epoch_num + 1, cfg.training.epochs, float(classification_loss), float(regression_loss),
                        np.mean(loss_hist)))

                del classification_loss
                del regression_loss
                pbar.update()
            except Exception as e:
                print(e)
                continue
        wandb.log({
            "Epoch": epoch_num + 1,
            "Claasification Loss": sum(epoch_classification_loss) / len(epoch_classification_loss),
            "Regression Loss": sum(epoch_regression_loss) / len(epoch_regression_loss),
            "Running Loss": sum(total_epoch_loss) / len(total_epoch_loss),
        })



        coco_eval.evaluate_coco(epoch_num, cfg, dataset_val, retinanet)
        scheduler.step(np.mean(epoch_loss))

        torch.save(retinanet, os.path.join(cfg.output.save_dir.classification.model_path, '{}_retinanet_{}.pt'.format(cfg.dataset.name, epoch_num)))

    retinanet.eval()
    torch.save(retinanet, os.path.join(cfg.output.save_dir.classification.model_path, 'model_final.pt'))

@hydra.main(config_path='configs', config_name='train')
def main(cfg):
    print(OmegaConf.to_yaml(cfg))
    train(cfg)

if __name__ == "__main__":
    main()
