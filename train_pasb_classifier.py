import argparse
import csv
import os
import time
import warnings

warnings.filterwarnings('ignore', message='Failed to load image Python extension.*')
import torch
from PIL import Image
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

from data.image_folder import make_dataset
from data.pasb_dataset import PASBDataset
from models import networks


class IHCLabelDataset(Dataset):
    def __init__(self, dataroot, phase, image_dir, label_file, load_size, crop_size,
                 max_dataset_size, is_train):
        self.root = os.path.join(dataroot, image_dir or phase + 'B')
        self.label_path = os.path.join(dataroot, label_file)
        if not os.path.exists(self.root):
            raise FileNotFoundError('IHC image directory not found: %s' % self.root)
        if not os.path.exists(self.label_path):
            raise FileNotFoundError('Label CSV not found: %s' % self.label_path)

        self.paths = sorted(make_dataset(self.root, max_dataset_size))
        self.path_index = self._build_path_index(self.paths, dataroot)
        self.samples = self._load_samples(max_dataset_size)
        if is_train:
            transform_list = [
                transforms.Resize(load_size, Image.BICUBIC),
                transforms.RandomCrop(crop_size),
                transforms.RandomHorizontalFlip(),
            ]
        else:
            transform_list = [
                transforms.Resize(load_size, Image.BICUBIC),
                transforms.CenterCrop(crop_size),
            ]
        transform_list += [
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ]
        self.transform = transforms.Compose(transform_list)

    def _build_path_index(self, paths, dataroot):
        index = {}
        for path in paths:
            keys = {
                path,
                os.path.abspath(path),
                os.path.basename(path),
                os.path.splitext(os.path.basename(path))[0],
            }
            for base in [self.root, dataroot]:
                try:
                    keys.add(os.path.relpath(path, base))
                except ValueError:
                    pass
            for key in keys:
                index[key] = path
        return index

    def _get_first_value(self, row, keys):
        for key in keys:
            value = row.get(key)
            if value not in [None, '']:
                return value
        return None

    def _validate_csv_columns(self, fieldnames):
        fieldnames = set(fieldnames or [])
        has_path = any(key in fieldnames for key in PASBDataset.PATH_KEYS)
        has_label = any(key in fieldnames for key in PASBDataset.LABEL_KEYS)
        if not has_path or not has_label:
            raise RuntimeError(
                'Classifier label CSV %s must contain a path column from [%s] and '
                'a label column from [%s]. Recommended columns are patt,label.' %
                (self.label_path, ', '.join(PASBDataset.PATH_KEYS), ', '.join(PASBDataset.LABEL_KEYS))
            )

    def _resolve_path(self, patt):
        candidates = [
            patt,
            os.path.normpath(patt),
            os.path.basename(patt),
            os.path.splitext(os.path.basename(patt))[0],
        ]
        for candidate in candidates:
            if candidate in self.path_index:
                return self.path_index[candidate]
        return None

    def _load_samples(self, max_dataset_size):
        samples = []
        skipped = 0
        with open(self.label_path, newline='') as csv_file:
            reader = csv.DictReader(csv_file)
            self._validate_csv_columns(reader.fieldnames)
            for row in reader:
                patt = self._get_first_value(row, PASBDataset.PATH_KEYS)
                label = self._get_first_value(row, PASBDataset.LABEL_KEYS)
                path = self._resolve_path(patt) if patt is not None else None
                if path is None or label is None:
                    skipped += 1
                    continue
                samples.append((path, int(label)))
                if len(samples) >= max_dataset_size:
                    break
        if not samples:
            raise RuntimeError('No labeled classifier samples matched files under %s.' % self.root)
        if skipped > 0:
            print('Skipped %d CSV rows without usable path/label or matching image files.' % skipped)
        return samples

    def __getitem__(self, index):
        path, label = self.samples[index]
        image = Image.open(path).convert('RGB')
        return self.transform(image), torch.tensor(label, dtype=torch.long), path

    def __len__(self):
        return len(self.samples)


def parse_args():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--dataroot', required=True, help='dataset root containing trainB and labels.csv')
    parser.add_argument('--name', default='ihc_classifier', help='checkpoint subdirectory name')
    parser.add_argument('--checkpoints_dir', default='./checkpoints', help='where classifier checkpoints are saved')
    parser.add_argument('--phase', default='train', help='dataset phase prefix used when --image_dir is empty')
    parser.add_argument('--image_dir', default='', help='IHC image directory relative to dataroot; defaults to <phase>B')
    parser.add_argument('--pasb_label_file', default='labels.csv', help='CSV file under dataroot')
    parser.add_argument('--pasb_classifier_net', default='resnet50',
                        choices=['small', 'resnet18', 'resnet50', 'convnext_tiny'])
    parser.add_argument('--pasb_classifier_pretrained_backbone', action='store_true',
                        help='use torchvision ImageNet weights when available')
    parser.add_argument('--pasb_num_classes', type=int, default=4)
    parser.add_argument('--input_nc', type=int, default=3)
    parser.add_argument('--ndf', type=int, default=64, help='width for the small classifier')
    parser.add_argument('--load_size', type=int, default=256)
    parser.add_argument('--crop_size', type=int, default=224)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--num_threads', type=int, default=4)
    parser.add_argument('--max_dataset_size', type=int, default=100000000)
    parser.add_argument('--n_epochs', type=int, default=20)
    parser.add_argument('--lr', type=float, default=0.0001)
    parser.add_argument('--beta1', type=float, default=0.9)
    parser.add_argument('--beta2', type=float, default=0.999)
    parser.add_argument('--print_freq', type=int, default=20)
    parser.add_argument('--save_epoch_freq', type=int, default=5)
    parser.add_argument('--device', default='auto', choices=['auto', 'cpu', 'cuda', 'mps'])
    parser.add_argument('--resume', default='', help='optional classifier checkpoint to resume from')
    return parser.parse_args()


def get_device(name):
    if name == 'auto':
        if torch.backends.mps.is_available():
            return torch.device('mps')
        if torch.cuda.is_available():
            return torch.device('cuda')
        return torch.device('cpu')
    if name == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('CUDA requested but not available')
    if name == 'mps' and not torch.backends.mps.is_available():
        raise RuntimeError('MPS requested but not available')
    return torch.device(name)


def load_state_dict(path, device):
    try:
        checkpoint = torch.load(path, map_location=str(device), weights_only=True)
    except TypeError:
        checkpoint = torch.load(path, map_location=str(device))
    if isinstance(checkpoint, dict):
        for key in ['state_dict', 'netC', 'model']:
            if key in checkpoint and isinstance(checkpoint[key], dict):
                checkpoint = checkpoint[key]
                break
    if any(key.startswith('module.') for key in checkpoint.keys()):
        checkpoint = {key.replace('module.', '', 1): value for key, value in checkpoint.items()}
    return checkpoint


def save_checkpoint(net, save_dir, label, opt):
    os.makedirs(save_dir, exist_ok=True)
    state = {
        'state_dict': net.state_dict(),
        'pasb_classifier_net': opt.pasb_classifier_net,
        'pasb_num_classes': opt.pasb_num_classes,
    }
    torch.save(state, os.path.join(save_dir, '%s_net_C.pth' % label))


def main():
    opt = parse_args()
    device = get_device(opt.device)
    dataset = IHCLabelDataset(opt.dataroot, opt.phase, opt.image_dir, opt.pasb_label_file,
                              opt.load_size, opt.crop_size, opt.max_dataset_size, is_train=True)
    dataloader = DataLoader(dataset, batch_size=opt.batch_size, shuffle=True,
                            num_workers=opt.num_threads, drop_last=False)

    net = networks.define_C(opt.input_nc, opt.ndf, opt.pasb_classifier_net, opt.pasb_num_classes,
                            pretrained_backbone=opt.pasb_classifier_pretrained_backbone)
    net.to(device)
    if opt.resume:
        missing, unexpected = net.load_state_dict(load_state_dict(opt.resume, device), strict=False)
        if missing or unexpected:
            print('Resume warning: missing keys=%d unexpected keys=%d' % (len(missing), len(unexpected)))

    criterion = nn.CrossEntropyLoss().to(device)
    optimizer = torch.optim.Adam(net.parameters(), lr=opt.lr, betas=(opt.beta1, opt.beta2))
    save_dir = os.path.join(opt.checkpoints_dir, opt.name)
    print('Training %s classifier on %d IHC images using %s' %
          (opt.pasb_classifier_net, len(dataset), device))

    total_iters = 0
    for epoch in range(1, opt.n_epochs + 1):
        start_time = time.time()
        total_loss = 0.0
        total_correct = 0
        total_seen = 0
        net.train()
        for images, labels, _ in dataloader:
            images = images.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()
            logits = net(images)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()

            total_iters += images.size(0)
            total_loss += loss.item() * images.size(0)
            total_correct += (logits.argmax(dim=1) == labels).sum().item()
            total_seen += images.size(0)
            if total_iters % opt.print_freq == 0:
                print('epoch %d, iters %d, loss %.4f, acc %.4f' %
                      (epoch, total_iters, total_loss / total_seen, total_correct / total_seen))

        print('End of epoch %d / %d, loss %.4f, acc %.4f, time %d sec' %
              (epoch, opt.n_epochs, total_loss / total_seen, total_correct / total_seen,
               time.time() - start_time))
        save_checkpoint(net, save_dir, 'latest', opt)
        if epoch % opt.save_epoch_freq == 0:
            save_checkpoint(net, save_dir, str(epoch), opt)


if __name__ == '__main__':
    main()
