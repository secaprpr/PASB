# pyrefly: ignore [missing-import]
import numpy as np
import os
import warnings
# pyrefly: ignore [missing-import]
import torch
# pyrefly: ignore [missing-import]
import torch.nn as nn
from .base_model import BaseModel
from . import networks
from .patchnce import PatchNCELoss
import util.util as util


class PASBModel(BaseModel):
    @staticmethod
    def modify_commandline_options(parser, is_train=True):
        """Configure options specific for PASB."""
        parser.add_argument('--mode', type=str, default="pasb", choices=['pasb', 'sb'])

        parser.add_argument('--lambda_GAN', type=float, default=1.0, help='weight for adversarial boundary loss')
        parser.add_argument('--lambda_NCE', type=float, default=1.0, help='weight for CUT/PatchNCE regularization')
        parser.add_argument('--lambda_SB', type=float, default=1.0, help='weight for entropy-regularized OT loss')
        parser.add_argument('--lambda_CDAL', type=float, default=1.0, help='weight for pathology prior alignment')
        parser.add_argument('--nce_idt', type=util.str2bool, nargs='?', const=True, default=True, help='use NCE loss for identity mapping')
        parser.add_argument('--nce_layers', type=str, default='0,4,8,12,16', help='compute NCE loss on which layers')
        parser.add_argument('--nce_includes_all_negatives_from_minibatch',
                            type=util.str2bool, nargs='?', const=True, default=False,
                            help='include negatives from other samples in the minibatch')
        parser.add_argument('--netF', type=str, default='mlp_sample', choices=['sample', 'reshape', 'mlp_sample'], help='how to downsample the feature map')
        parser.add_argument('--netF_nc', type=int, default=256)
        parser.add_argument('--nce_T', type=float, default=0.07, help='temperature for NCE loss')
        parser.add_argument('--num_patches', type=int, default=256, help='number of patches per layer')
        parser.add_argument('--flip_equivariance',
                            type=util.str2bool, nargs='?', const=True, default=False,
                            help='enforce flip-equivariance as additional regularization')

        parser.add_argument('--pasb_num_classes', type=int, default=4, help='number of IRS/pathology prior classes')
        parser.add_argument('--pasb_classifier_net', type=str, default='resnet50',
                            choices=['small', 'resnet18', 'resnet50', 'convnext_tiny'],
                            help='pathology classifier architecture for CDAL')
        parser.add_argument('--pasb_classifier_pretrained_backbone', type=util.str2bool, nargs='?', const=True,
                            default=False, help='initialize torchvision classifier backbone from ImageNet weights')
        parser.add_argument('--pasb_alpha', type=float, default=0.1, help='SDPR gradient correction strength')
        parser.add_argument('--pasb_beta', type=float, default=1.0, help='SDPR adaptive noise decay')
        parser.add_argument('--pasb_use_sdpr', type=util.str2bool, nargs='?', const=True, default=True, help='enable SDPR sampling')
        parser.add_argument('--pasb_pretrained_C', type=str, default='./checkpoints/ihc_classifier/latest_net_C.pth',
                            help='optional pretrained pathology classifier checkpoint')
        parser.add_argument('--pasb_update_C', type=util.str2bool, nargs='?', const=True, default=False,
                            help='update pathology classifier during PASB training; false freezes a loaded classifier')
        parser.add_argument('--pasb_label_keys', type=str, default='label,y,irs_label,pathology_label',
                            help='comma-separated input keys used for pathology labels')

        parser.set_defaults(pool_size=0, nce_idt=True, lambda_NCE=1.0, dataset_mode='pasb')
        return parser

    def __init__(self, opt):
        BaseModel.__init__(self, opt)

        self.loss_names = ['G_GAN', 'D_real', 'D_fake', 'G', 'NCE', 'SB', 'CDAL']
        self.visual_names = ['real_A', 'real_A_noisy', 'fake_B', 'real_B']
        if self.opt.phase == 'test':
            self.visual_names = ['real']
            for NFE in range(self.opt.num_timesteps):
                self.visual_names.append('fake_' + str(NFE + 1))
        self.nce_layers = [int(i) for i in self.opt.nce_layers.split(',')]
        self.label_keys = [key.strip() for key in self.opt.pasb_label_keys.split(',')]

        if self.isTrain:
            self.loss_names += ['C']
        if opt.nce_idt and self.isTrain:
            self.loss_names += ['NCE_Y']
            self.visual_names += ['idt_B']

        if self.isTrain:
            self.model_names = ['G', 'F', 'D', 'E', 'C']
        else:
            self.model_names = ['G']

        self.netG = networks.define_G(opt.input_nc, opt.output_nc, opt.ngf, opt.netG, opt.normG, not opt.no_dropout,
                                      opt.init_type, opt.init_gain, opt.no_antialias, opt.no_antialias_up,
                                      self.gpu_ids, opt)
        self.netF = networks.define_F(opt.input_nc, opt.netF, opt.normG, not opt.no_dropout, opt.init_type,
                                      opt.init_gain, opt.no_antialias, self.gpu_ids, opt)

        if self.isTrain:
            self.netD = networks.define_D(opt.output_nc, opt.ndf, opt.netD, opt.n_layers_D, opt.normD,
                                          opt.init_type, opt.init_gain, opt.no_antialias, self.gpu_ids, opt)
            self.netE = networks.define_D(opt.output_nc * 4, opt.ndf, opt.netD, opt.n_layers_D, opt.normD,
                                          opt.init_type, opt.init_gain, opt.no_antialias, self.gpu_ids, opt)
            self.netC = networks.define_C(opt.output_nc, opt.ndf, opt.pasb_classifier_net, opt.pasb_num_classes,
                                          opt.init_type, opt.init_gain, self.gpu_ids,
                                          opt.pasb_classifier_pretrained_backbone)
            self.update_C = opt.pasb_update_C
            if opt.pasb_pretrained_C and os.path.isfile(opt.pasb_pretrained_C):
                self._load_classifier(opt.pasb_pretrained_C)
            else:
                self.update_C = True
                warnings.warn(
                    'PASB pretrained pathology classifier was not found. '
                    'Training netC online for connectivity testing; provide '
                    '--pasb_pretrained_C and keep --pasb_update_C false for reproduction.',
                    RuntimeWarning
                )

            self.criterionGAN = networks.GANLoss(opt.gan_mode).to(self.device)
            self.criterionNCE = []
            for nce_layer in self.nce_layers:
                self.criterionNCE.append(PatchNCELoss(opt).to(self.device))
            self.criterionCE = nn.CrossEntropyLoss().to(self.device)

            self.optimizer_G = torch.optim.Adam(self.netG.parameters(), lr=opt.lr, betas=(opt.beta1, opt.beta2))
            self.optimizer_D = torch.optim.Adam(self.netD.parameters(), lr=opt.lr, betas=(opt.beta1, opt.beta2))
            self.optimizer_E = torch.optim.Adam(self.netE.parameters(), lr=opt.lr, betas=(opt.beta1, opt.beta2))
            self.optimizers.append(self.optimizer_G)
            self.optimizers.append(self.optimizer_D)
            self.optimizers.append(self.optimizer_E)
            if self.update_C:
                self.optimizer_C = torch.optim.Adam(self.netC.parameters(), lr=opt.lr, betas=(opt.beta1, opt.beta2))
                self.optimizers.append(self.optimizer_C)
            else:
                self.set_requires_grad(self.netC, False)

    def _load_classifier(self, checkpoint_path):
        state_dict = self._torch_load(checkpoint_path)
        if isinstance(state_dict, dict):
            for key in ['state_dict', 'netC', 'model']:
                if key in state_dict and isinstance(state_dict[key], dict):
                    state_dict = state_dict[key]
                    break
        if any(key.startswith('module.') for key in state_dict.keys()):
            state_dict = {key.replace('module.', '', 1): value for key, value in state_dict.items()}
        missing, unexpected = self.netC.load_state_dict(state_dict, strict=False)
        if missing or unexpected:
            warnings.warn(
                'Loaded PASB classifier with missing keys (%d) and unexpected keys (%d). '
                'Check --pasb_classifier_net matches the checkpoint architecture.' %
                (len(missing), len(unexpected)),
                RuntimeWarning
            )

    def data_dependent_initialize(self, data, data2):
        bs_per_gpu = data["A"].size(0) // max(len(self.opt.gpu_ids), 1)
        self.set_input(data, data2)
        self.real_A = self.real_A[:bs_per_gpu]
        self.real_B = self.real_B[:bs_per_gpu]
        self.real_A2 = self.real_A2[:bs_per_gpu]
        self.real_B2 = self.real_B2[:bs_per_gpu]
        if self.opt.isTrain:
            self.pathology_label = self.pathology_label[:bs_per_gpu]
        self.forward()
        if self.opt.isTrain:
            self.set_requires_grad(self.netC, False)
            self.compute_G_loss().backward()
            if self.update_C:
                self.set_requires_grad(self.netC, True)
            self.compute_D_loss().backward()
            self.compute_E_loss().backward()
            if self.update_C:
                self.compute_C_loss().backward()
            if self.opt.lambda_NCE > 0.0:
                self.optimizer_F = torch.optim.Adam(self.netF.parameters(), lr=self.opt.lr,
                                                    betas=(self.opt.beta1, self.opt.beta2))
                self.optimizers.append(self.optimizer_F)

    def optimize_parameters(self):
        self.forward()
        self.netG.train()
        self.netE.train()
        self.netD.train()
        self.netF.train()
        if self.update_C:
            self.netC.train()
        else:
            self.netC.eval()

        self.set_requires_grad(self.netD, True)
        self.optimizer_D.zero_grad()
        self.loss_D = self.compute_D_loss()
        self.loss_D.backward()
        self.optimizer_D.step()

        self.set_requires_grad(self.netE, True)
        self.optimizer_E.zero_grad()
        self.loss_E = self.compute_E_loss()
        self.loss_E.backward()
        self.optimizer_E.step()

        if self.update_C:
            self.set_requires_grad(self.netC, True)
            self.optimizer_C.zero_grad()
            self.loss_C = self.compute_C_loss()
            self.loss_C.backward()
            self.optimizer_C.step()
        else:
            with torch.no_grad():
                self.loss_C = self.compute_C_loss()

        self.set_requires_grad(self.netD, False)
        self.set_requires_grad(self.netE, False)
        self.set_requires_grad(self.netC, False)

        self.optimizer_G.zero_grad()
        if self.opt.netF == 'mlp_sample':
            self.optimizer_F.zero_grad()
        self.loss_G = self.compute_G_loss()
        self.loss_G.backward()
        self.optimizer_G.step()
        if self.opt.netF == 'mlp_sample':
            self.optimizer_F.step()
        if self.update_C:
            self.set_requires_grad(self.netC, True)

    def set_input(self, input, input2=None):
        AtoB = self.opt.direction == 'AtoB'
        self.real_A = input['A' if AtoB else 'B'].to(self.device)
        self.real_B = input['B' if AtoB else 'A'].to(self.device)
        if input2 is not None:
            self.real_A2 = input2['A' if AtoB else 'B'].to(self.device)
            self.real_B2 = input2['B' if AtoB else 'A'].to(self.device)
        else:
            self.real_A2 = self.real_A
            self.real_B2 = self.real_B
        if self.isTrain:
            self.pathology_label = self._get_pathology_label(input, self.real_B)
        self.image_paths = input['A_paths' if AtoB else 'B_paths']

    def _get_pathology_label(self, input, reference):
        for key in self.label_keys:
            if key in input:
                label = input[key]
                if not torch.is_tensor(label):
                    label = torch.tensor(label)
                return label.to(self.device).long().reshape(-1)
        return self._estimate_irs_label(reference)

    def _estimate_irs_label(self, image):
        """Fallback four-level IRS-like prior from normalized IHC color statistics."""
        x = (image.detach() + 1.0) * 0.5
        r, g, b = x[:, 0], x[:, 1], x[:, 2]
        brownness = (r - 0.5 * (g + b)).clamp(min=0.0)
        area = (brownness > 0.08).float().mean(dim=(1, 2))
        intensity = brownness.mean(dim=(1, 2))
        score = area * intensity
        label = torch.zeros_like(score, dtype=torch.long)
        label[score > 0.015] = 1
        label[score > 0.035] = 2
        label[score > 0.070] = 3
        return label.to(self.device)

    def _make_times(self):
        T = self.opt.num_timesteps
        incs = np.array([0] + [1 / (i + 1) for i in range(T - 1)])
        times = np.cumsum(incs)
        times = times / times[-1]
        times = 0.5 * times[-1] + 0.5 * times
        times = np.concatenate([np.zeros(1), times])
        return torch.tensor(times).float().to(self.device)

    def _log_similarity(self, pred, ref):
        """PASB Eq. 11 style similarity using image-level mean and variance."""
        pred_mu = pred.mean(dim=(1, 2, 3))
        ref_mu = ref.mean(dim=(1, 2, 3))
        pred_var = pred.flatten(1).var(dim=1, unbiased=False)
        ref_var = ref.flatten(1).var(dim=1, unbiased=False)
        return -((pred_mu - ref_mu) ** 2 + (pred_var - ref_var) ** 2)

    def _sdpr_refine(self, pred, ref):
        if not self.opt.pasb_use_sdpr:
            score = torch.ones(pred.size(0), 1, 1, 1, device=pred.device)
            return pred.detach(), score
        with torch.enable_grad():
            target = pred.detach().requires_grad_(True)
            log_s = self._log_similarity(target, ref.detach()).sum()
            grad = torch.autograd.grad(log_s, target, retain_graph=False, create_graph=False)[0]
            refined = (target + self.opt.pasb_alpha * grad).detach().clamp(-1.0, 1.0)
        score = torch.exp(self._log_similarity(refined, ref.detach())).reshape(-1, 1, 1, 1)
        return refined, score.clamp(min=1e-4, max=1.0)

    def _bridge_step(self, xt, pred, ref, times, t):
        delta = times[t] - times[t - 1]
        denom = times[-1] - times[t - 1]
        inter = (delta / denom).reshape(-1, 1, 1, 1)
        scale = (delta * (1 - delta / denom)).reshape(-1, 1, 1, 1)
        refined, score = self._sdpr_refine(pred, ref)
        variance = scale * self.opt.tau
        if self.opt.pasb_use_sdpr:
            variance = variance * torch.exp(-self.opt.pasb_beta * score)
        noise = variance.sqrt() * torch.randn_like(xt).to(xt.device)
        return (1 - inter) * xt + inter * refined + noise

    def _sample_path(self, start, ref, times, stop_idx):
        xt = start
        xt_1 = start
        for t in range(stop_idx + 1):
            if t > 0:
                xt = self._bridge_step(xt, xt_1.detach(), ref, times, t)
            time_idx = (t * torch.ones(size=[start.shape[0]]).to(start.device)).long()
            z = torch.randn(size=[start.shape[0], 4 * self.opt.ngf]).to(start.device)
            xt_1 = self.netG(xt, time_idx, z)
        return xt.detach(), xt_1.detach()

    def _forward_test(self, times):
        tau = self.opt.tau
        self.real = self.real_A
        xt = self.real_A
        xt_1 = self.real_A
        with torch.no_grad():
            self.netG.eval()
            for t in range(self.opt.num_timesteps):
                if t > 0:
                    delta = times[t] - times[t - 1]
                    denom = times[-1] - times[t - 1]
                    inter = (delta / denom).reshape(-1, 1, 1, 1)
                    scale = (delta * (1 - delta / denom)).reshape(-1, 1, 1, 1)
                    noise = (scale * tau).sqrt() * torch.randn_like(xt).to(self.real_A.device)
                    xt = (1 - inter) * xt + inter * xt_1.detach() + noise
                time_idx = (t * torch.ones(size=[self.real_A.shape[0]]).to(self.real_A.device)).long()
                z = torch.randn(size=[self.real_A.shape[0], 4 * self.opt.ngf]).to(self.real_A.device)
                xt_1 = self.netG(xt, time_idx, z)
                setattr(self, 'fake_' + str(t + 1), xt_1)

    def forward(self):
        times = self._make_times()
        self.times = times
        if self.opt.phase == 'test':
            self._forward_test(times)
            return

        bs = self.real_A.size(0)
        t = torch.randint(self.opt.num_timesteps, size=[1]).to(self.device).long().item()
        self.time_idx = torch.full((bs,), t, device=self.device, dtype=torch.long)
        self.timestep = times[self.time_idx]

        with torch.no_grad():
            self.netG.eval()
            self.real_A_noisy, _ = self._sample_path(self.real_A, self.real_B, times, t)
            self.real_A_noisy2, _ = self._sample_path(self.real_A2, self.real_B2, times, t)
            if self.opt.nce_idt:
                self.XtB, _ = self._sample_path(self.real_B, self.real_B, times, t)

        self.real = torch.cat((self.real_A, self.real_B), dim=0) if self.opt.nce_idt and self.opt.isTrain else self.real_A
        self.realt = torch.cat((self.real_A_noisy, self.XtB), dim=0) if self.opt.nce_idt and self.opt.isTrain else self.real_A_noisy
        z_in = torch.randn(size=[self.realt.size(0), 4 * self.opt.ngf]).to(self.real_A.device)
        z_in2 = torch.randn(size=[self.real_A_noisy2.size(0), 4 * self.opt.ngf]).to(self.real_A.device)

        if self.opt.flip_equivariance:
            self.flipped_for_equivariance = self.opt.isTrain and (np.random.random() < 0.5)
            if self.flipped_for_equivariance:
                self.real = torch.flip(self.real, [3])
                self.realt = torch.flip(self.realt, [3])

        realt_time_idx = torch.full((self.realt.size(0),), t, device=self.device, dtype=torch.long)
        self.fake = self.netG(self.realt, realt_time_idx, z_in)
        self.fake_B2 = self.netG(self.real_A_noisy2, self.time_idx, z_in2)
        self.fake_B = self.fake[:self.real_A.size(0)]
        if self.opt.nce_idt:
            self.idt_B = self.fake[self.real_A.size(0):]

    def compute_D_loss(self):
        fake = self.fake_B.detach()
        pred_fake = self.netD(fake, self.time_idx)
        self.loss_D_fake = self.criterionGAN(pred_fake, False).mean()
        self.pred_real = self.netD(self.real_B, self.time_idx)
        loss_D_real = self.criterionGAN(self.pred_real, True)
        self.loss_D_real = loss_D_real.mean()
        self.loss_D = (self.loss_D_fake + self.loss_D_real) * 0.5
        return self.loss_D

    def compute_E_loss(self):
        XtXt_1 = torch.cat([self.real_A_noisy, self.fake_B.detach()], dim=1)
        XtXt_2 = torch.cat([self.real_A_noisy2, self.fake_B2.detach()], dim=1)
        temp = torch.logsumexp(self.netE(XtXt_1, self.time_idx, XtXt_2).reshape(-1), dim=0).mean()
        self.loss_E = -self.netE(XtXt_1, self.time_idx, XtXt_1).mean() + temp + temp ** 2
        return self.loss_E

    def compute_C_loss(self):
        pred_real = self.netC(self.real_B)
        self.loss_C = self.criterionCE(pred_real, self.pathology_label)
        return self.loss_C

    def compute_G_loss(self):
        fake = self.fake_B

        if self.opt.lambda_GAN > 0.0:
            pred_fake = self.netD(fake, self.time_idx)
            self.loss_G_GAN = self.criterionGAN(pred_fake, True).mean() * self.opt.lambda_GAN
        else:
            self.loss_G_GAN = 0.0

        self.loss_SB = 0
        if self.opt.lambda_SB > 0.0:
            XtXt_1 = torch.cat([self.real_A_noisy, self.fake_B], dim=1)
            XtXt_2 = torch.cat([self.real_A_noisy2, self.fake_B2], dim=1)
            ET_XY = self.netE(XtXt_1, self.time_idx, XtXt_1).mean() - \
                torch.logsumexp(self.netE(XtXt_1, self.time_idx, XtXt_2).reshape(-1), dim=0)
            self.loss_SB = -(self.opt.num_timesteps - self.time_idx[0]) / self.opt.num_timesteps * self.opt.tau * ET_XY
            self.loss_SB += self.opt.tau * torch.mean((self.real_A_noisy - self.fake_B) ** 2)

        if self.opt.lambda_CDAL > 0.0:
            logits_fake = self.netC(fake)
            self.loss_CDAL = self.criterionCE(logits_fake, self.pathology_label) * self.opt.lambda_CDAL
        else:
            self.loss_CDAL = 0.0

        if self.opt.lambda_NCE > 0.0:
            self.loss_NCE = self.calculate_NCE_loss(self.real_A, fake)
        else:
            self.loss_NCE = 0.0

        if self.opt.nce_idt and self.opt.lambda_NCE > 0.0:
            self.loss_NCE_Y = self.calculate_NCE_loss(self.real_B, self.idt_B)
            loss_NCE_both = (self.loss_NCE + self.loss_NCE_Y) * 0.5
        else:
            loss_NCE_both = self.loss_NCE

        # calculate_NCE_loss returns the raw PatchNCE value; apply lambda_NCE exactly once here.
        self.loss_G = self.loss_G_GAN + self.opt.lambda_SB * self.loss_SB + \
            self.loss_CDAL + self.opt.lambda_NCE * loss_NCE_both
        return self.loss_G

    def calculate_NCE_loss(self, src, tgt):
        n_layers = len(self.nce_layers)
        time_idx = torch.zeros(size=[src.size(0)], device=src.device, dtype=torch.long)
        z = torch.randn(size=[src.size(0), 4 * self.opt.ngf]).to(src.device)
        feat_q = self.netG(tgt, time_idx, z, self.nce_layers, encode_only=True)

        if self.opt.flip_equivariance and self.flipped_for_equivariance:
            feat_q = [torch.flip(fq, [3]) for fq in feat_q]

        feat_k = self.netG(src, time_idx, z, self.nce_layers, encode_only=True)
        feat_k_pool, sample_ids = self.netF(feat_k, self.opt.num_patches, None)
        feat_q_pool, _ = self.netF(feat_q, self.opt.num_patches, sample_ids)

        total_nce_loss = 0.0
        for f_q, f_k, crit, nce_layer in zip(feat_q_pool, feat_k_pool, self.criterionNCE, self.nce_layers):
            loss = crit(f_q, f_k)
            total_nce_loss += loss.mean()

        return total_nce_loss / n_layers
