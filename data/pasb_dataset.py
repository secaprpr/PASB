import csv
import os.path
import warnings
from data.base_dataset import BaseDataset, get_params, get_transform
from data.image_folder import make_dataset
from PIL import Image
import util.util as util


class PASBDataset(BaseDataset):
    """
    Load weakly supervised H&E/IHC adjacent-section pairs for PASB.

    It follows the CUT folder convention:
        dataroot/trainA, dataroot/trainB, dataroot/testA

    Unlike UnalignedDataset, B is selected with the same index as A so the
    reference IHC image can provide CDAL and SDPR guidance during training.
    At test time, B is optional because inference is H&E-only.
    Optionally, a labels.csv file under dataroot may provide IRS labels with
    columns: patt,label. The patt value can be a basename or a relative path.
    When the CSV exists, only labeled rows from the CSV are loaded.
    """

    PATH_KEYS = ('patt', 'path', 'file', 'filename', 'image')
    LABEL_KEYS = ('label', 'y', 'irs_label', 'pathology_label')

    @staticmethod
    def modify_commandline_options(parser, is_train):
        parser.add_argument('--pasb_label_file', type=str, default='labels.csv',
                            help='optional CSV file under dataroot with columns patt,label')
        return parser

    def __init__(self, opt):
        BaseDataset.__init__(self, opt)
        self.dir_A = os.path.join(opt.dataroot, opt.phase + 'A')
        self.dir_B = os.path.join(opt.dataroot, opt.phase + 'B')

        if opt.phase == "test" and not os.path.exists(self.dir_A) \
           and os.path.exists(os.path.join(opt.dataroot, "valA")):
            self.dir_A = os.path.join(opt.dataroot, "valA")
            self.dir_B = os.path.join(opt.dataroot, "valB")
        self.has_reference = os.path.exists(self.dir_B)

        self.A_paths = sorted(make_dataset(self.dir_A, opt.max_dataset_size))
        self.B_paths = sorted(make_dataset(self.dir_B, opt.max_dataset_size)) if self.has_reference else self.A_paths
        self.samples = self._load_labeled_samples()
        if self.samples:
            self.A_paths = [sample['A_path'] for sample in self.samples]
            self.B_paths = [sample['B_path'] for sample in self.samples]
        self.A_size = len(self.A_paths)
        self.B_size = len(self.B_paths)
        self.labels = self._load_labels() if not self.samples else {}

    def _label_path(self):
        label_path = os.path.join(self.opt.dataroot, self.opt.pasb_label_file)
        return label_path

    def _get_first_value(self, row, keys):
        for key in keys:
            value = row.get(key)
            if value not in [None, '']:
                return value
        return None

    def _validate_csv_columns(self, fieldnames, label_path):
        fieldnames = set(fieldnames or [])
        has_path = any(key in fieldnames for key in self.PATH_KEYS)
        has_label = any(key in fieldnames for key in self.LABEL_KEYS)
        missing = []
        if not has_path:
            missing.append('one path column from [%s]' % ', '.join(self.PATH_KEYS))
        if not has_label:
            missing.append('one label column from [%s]' % ', '.join(self.LABEL_KEYS))
        if missing:
            raise RuntimeError(
                'PASB label CSV %s is missing %s. Recommended columns are patt,label.' %
                (label_path, ' and '.join(missing))
            )

    def _load_labels(self):
        label_path = self._label_path()
        labels = {}
        if not os.path.exists(label_path):
            return labels
        with open(label_path, newline='') as csv_file:
            reader = csv.DictReader(csv_file)
            self._validate_csv_columns(reader.fieldnames, label_path)
            for row in reader:
                path = self._get_first_value(row, self.PATH_KEYS)
                label = self._get_first_value(row, self.LABEL_KEYS)
                if path is None or label is None:
                    continue
                labels[path] = int(label)
                labels[os.path.basename(path)] = int(label)
        return labels

    def _build_path_index(self, paths, root):
        index = {}
        for path in paths:
            keys = {
                path,
                os.path.abspath(path),
                os.path.basename(path),
                os.path.splitext(os.path.basename(path))[0],
            }
            for base in [root, self.opt.dataroot]:
                try:
                    keys.add(os.path.relpath(path, base))
                except ValueError:
                    pass
            for key in keys:
                index[key] = path
        return index

    def _resolve_path(self, patt, index):
        candidates = [
            patt,
            os.path.normpath(patt),
            os.path.basename(patt),
            os.path.splitext(os.path.basename(patt))[0],
        ]
        for candidate in candidates:
            if candidate in index:
                return index[candidate]
        return None

    def _load_labeled_samples(self):
        label_path = self._label_path()
        if not os.path.exists(label_path):
            return []

        A_index = self._build_path_index(self.A_paths, self.dir_A)
        B_index = self._build_path_index(self.B_paths, self.dir_B) if self.has_reference else {}
        samples = []
        malformed = 0
        with open(label_path, newline='') as csv_file:
            reader = csv.DictReader(csv_file)
            self._validate_csv_columns(reader.fieldnames, label_path)
            for row in reader:
                patt = self._get_first_value(row, self.PATH_KEYS)
                label = self._get_first_value(row, self.LABEL_KEYS)
                if patt is None or label is None:
                    malformed += 1
                    continue
                A_path = self._resolve_path(patt, A_index)
                B_path = self._resolve_path(patt, B_index) if self.has_reference else A_path
                if A_path is None or B_path is None:
                    continue
                samples.append({'A_path': A_path, 'B_path': B_path, 'label': int(label)})
                if len(samples) >= self.opt.max_dataset_size:
                    break

        if not samples:
            raise RuntimeError(
                'PASB label CSV was found, but no rows matched images in the current phase. '
                'Check that the patt column matches files under %s%s.' %
                (self.dir_A, (' and %s' % self.dir_B) if self.has_reference else '')
            )
        if malformed > 0:
            warnings.warn(
                'PASB label CSV skipped %d rows without usable path/label values.' % malformed,
                RuntimeWarning
            )
        return samples

    def __getitem__(self, index):
        A_path = self.A_paths[index % self.A_size]
        B_path = self.B_paths[index % self.B_size]
        A_img = Image.open(A_path).convert('RGB')
        B_img = Image.open(B_path).convert('RGB') if self.has_reference else A_img.copy()

        is_finetuning = self.opt.isTrain and self.current_epoch > self.opt.n_epochs
        modified_opt = util.copyconf(self.opt, load_size=self.opt.crop_size if is_finetuning else self.opt.load_size)
        transform_params = get_params(modified_opt, A_img.size)
        transform = get_transform(modified_opt, transform_params)
        A = transform(A_img)
        B = transform(B_img)

        data = {'A': A, 'B': B, 'A_paths': A_path, 'B_paths': B_path}
        if self.samples:
            label = self.samples[index % len(self.samples)]['label']
        else:
            label = self.labels.get(B_path, self.labels.get(os.path.basename(B_path)))
        if label is not None:
            data['label'] = label
        return data

    def __len__(self):
        return max(self.A_size, self.B_size)
