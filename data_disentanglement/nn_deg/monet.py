"""C. P. Burgess et al., "MONet: Unsupervised Scene Decomposition and Representation," pp. 1–22, 2019."""
from itertools import chain
import torch.nn.functional as F
import torch
from torch import nn, optim
from torch.nn import init


def init_weights(net, init_type='normal', init_gain=0.02):
    """Initialize network weights.

    Parameters:
        net (network)   -- network to be initialized
        init_type (str) -- the name of an initialization method: normal | xavier | kaiming | orthogonal
        init_gain (float)    -- scaling factor for normal, xavier and orthogonal.

    We use 'normal' in the original pix2pix and CycleGAN paper. But xavier and kaiming might
    work better for some applications. Feel free to try yourself.
    """

    def init_func(m):  # define the initialization function
        classname = m.__class__.__name__
        if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
            if init_type == 'normal':
                init.normal_(m.weight.data, 0.0, init_gain)
            elif init_type == 'xavier':
                init.xavier_normal_(m.weight.data, gain=init_gain)
            elif init_type == 'kaiming':
                init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
            elif init_type == 'orthogonal':
                init.orthogonal_(m.weight.data, gain=init_gain)
            else:
                raise NotImplementedError('initialization method [%s] is not implemented' % init_type)
            if hasattr(m, 'bias') and m.bias is not None:
                init.constant_(m.bias.data, 0.0)
        elif classname.find(
                'BatchNorm2d') != -1:  # BatchNorm Layer's weight is not a matrix; only normal distribution applies.
            init.normal_(m.weight.data, 1.0, init_gain)
            init.constant_(m.bias.data, 0.0)

    print('initialize network with %s' % init_type)
    net.apply(init_func)  # apply the initialization function <init_func>


class Flatten(nn.Module):

    def forward(self, x):
        return x.flatten(start_dim=1)


class MONetModel():

    def __init__(self, num_slots, z_dim, lr, device, input_nc=3, beta=0.5, gamma=0.5):
        """Initialize this model class.

        A few things can be done here.
        - (required) call the initialization function of BaseModel
        - define loss function, visualization images, model names, and optimizers
        """
        self.beta = beta
        self.gamma = gamma
        self.num_slots = num_slots
        self.input_nc = input_nc
        self.z_dim = z_dim
        self.device = device
        self.lr = lr

        self.loss_D = None
        self.loss_E = None
        self.loss_mask = None

        self.loss_names = ['E', 'D', 'mask']
        self.visual_names = ['m{}'.format(i) for i in range(self.num_slots)] + \
                            ['tm{}'.format(i) for i in range(self.num_slots)] + \
                            ['x{}'.format(i) for i in range(self.num_slots)] + \
                            ['xm{}'.format(i) for i in range(self.num_slots)] + \
                            ['xtm{}'.format(i) for i in range(self.num_slots)]
        # ['x', 'x_tilde']
        self.model_names = ['Attn', 'CVAE']
        self.netAttn = Attention(self.input_nc, 1).to(self.device)
        init_weights(self.netAttn)
        self.netCVAE = ComponentVAE(self.input_nc, self.z_dim).to(self.device)
        init_weights(self.netCVAE)

        self.criterionKL = nn.KLDivLoss(reduction='batchmean')
        self.optimizer = optim.RMSprop(chain(self.netAttn.parameters(), self.netCVAE.parameters()), lr=self.lr)
        self.optimizers = [self.optimizer]

    def forward(self, x):
        """Run forward pass. This will be called by both functions <optimize_parameters> and <test>."""
        self.loss_E = 0
        self.x_tilde = 0
        b = []
        m = []
        m_tilde_logits = []
        x_mu = []
        z_mu = []

        # Initial s_k = 1: shape = (N, 1, H, W)
        shape = list(x.shape)
        shape[1] = 1
        log_s_k = x.new_zeros(shape)

        for k in range(self.num_slots):
            # Derive mask from current scope
            if k != self.num_slots - 1:
                log_alpha_k, alpha_logits_k = self.netAttn(x, log_s_k)
                log_m_k = log_s_k + log_alpha_k
                # Compute next scope
                log_s_k += -alpha_logits_k + log_alpha_k
            else:
                log_m_k = log_s_k

            # Get component and mask reconstruction, as well as the z_k parameters
            m_tilde_k_logits, x_mu_k, x_logvar_k, z_mu_k, z_logvar_k = self.netCVAE(x, log_m_k, k == 0)
            x_mu.append(x_mu_k)
            z_mu.append(z_mu_k)

            # KLD is additive for independent distributions
            self.loss_E += -0.5 * (1 + z_logvar_k - z_mu_k.pow(2) - z_logvar_k.exp()).sum()

            m_k = log_m_k.exp()
            x_k_masked = m_k * x_mu_k

            # Exponents for the decoder loss
            b_k = log_m_k - 0.5 * x_logvar_k - (x - x_mu_k).pow(2) / (2 * x_logvar_k.exp())
            b.append(b_k.unsqueeze(1))

            # Get outputs for kth step
            # setattr(self, 'm{}'.format(k), m_k * 2. - 1.) # shift mask from [0, 1] to [-1, 1]
            setattr(self, 'm{}'.format(k), m_k)
            setattr(self, 'x{}'.format(k), x_mu_k)
            setattr(self, 'xm{}'.format(k), x_k_masked)

            # Iteratively reconstruct the output image
            self.x_tilde += x_k_masked
            # Accumulate
            m.append(m_k)
            m_tilde_logits.append(m_tilde_k_logits)

        # print(b[0].size())
        self.b = torch.cat(b, dim=1)
        self.m = torch.cat(m, dim=1)
        self.m_tilde_logits = torch.cat(m_tilde_logits, dim=1)
        self.x_mu = torch.stack(x_mu, dim=1)
        self.z_mu = torch.stack(z_mu, dim=1)

    def backward(self, x):
        """Calculate losses, gradients, and update network weights; called in every training iteration"""
        n = x.shape[0]
        self.loss_E /= n
        self.loss_D = -torch.logsumexp(self.b, dim=1).sum() / n
        # self.loss_D = self.b.sum() / n
        self.m_tilde = self.m_tilde_logits.softmax(dim=1)
        # self.m_tilde = self.m_tilde_logits.log_softmax(dim=1)

        for k in range(self.num_slots):
            # print(self.m_tilde[:, k, :, :].unsqueeze(dim=1).size())
            mt = self.m_tilde[:, k, :, :].unsqueeze(dim=1)
            x_k_tilde_masked = mt * self.x_mu[:, k, :, :, :]
            setattr(self, 'xtm{}'.format(k), x_k_tilde_masked)
            setattr(self, 'tm{}'.format(k), mt)
        # print(torch.sum(self.m_tilde, dim=1))
        # print(torch.sum(self.m, dim=1))
        self.loss_mask = self.criterionKL(self.m_tilde.log(), self.m)
        loss = self.loss_D + self.beta * self.loss_E + self.gamma * self.loss_mask
        # loss = self.loss_D
        loss.backward()
        return loss.detach().item()

    def optimize_parameters(self, x):
        """Update network weights; it will be called in every training iteration."""
        self.forward(x)  # first call forward to calculate intermediate results
        self.optimizer.zero_grad()  # clear network G's existing gradients
        loss = self.backward(x)  # calculate gradients for network G
        self.optimizer.step()  # update gradients for network G
        return loss


class ComponentVAE(nn.Module):

    def __init__(self, input_nc, z_dim=16, full_res=True):
        super().__init__()
        self._input_nc = input_nc
        self._z_dim = z_dim
        # full_res = False # full res: 128x128, low res: 64x64
        h_dim = 4096 if full_res else 1024
        self.encoder = nn.Sequential(
            nn.Conv2d(input_nc + 1, 32, 3, stride=2, padding=1),
            nn.ReLU(True),
            nn.Conv2d(32, 32, 3, stride=2, padding=1),
            nn.ReLU(True),
            nn.Conv2d(32, 64, 3, stride=2, padding=1),
            nn.ReLU(True),
            nn.Conv2d(64, 64, 3, stride=2, padding=1),
            nn.ReLU(True),
            Flatten(),
            nn.Linear(h_dim, 256),
            nn.ReLU(True),
            nn.Linear(256, 32)
        )
        self.decoder = nn.Sequential(
            nn.Conv2d(z_dim + 2, 32, 3),
            nn.ReLU(True),
            nn.Conv2d(32, 32, 3),
            nn.ReLU(True),
            nn.Conv2d(32, 32, 3),
            nn.ReLU(True),
            nn.Conv2d(32, 32, 3),
            nn.ReLU(True),
            nn.Conv2d(32, input_nc + 1, 1),
        )
        self._bg_logvar = 2 * torch.tensor(0.09).log()
        self._fg_logvar = 2 * torch.tensor(0.11).log()
        # self._bg_logvar = 2 * torch.tensor(0.0001).log()
        # self._fg_logvar = 2 * torch.tensor(0.01).log()

    @staticmethod
    def reparameterize(mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(mu)
        return mu + eps * std

    @staticmethod
    def spatial_broadcast(z, h, w):
        # Batch size
        n = z.shape[0]
        # Expand spatially: (n, z_dim) -> (n, z_dim, h, w)
        z_b = z.view((n, -1, 1, 1)).expand(-1, -1, h, w)
        # Coordinate axes:
        x = torch.linspace(-1, 1, w, device=z.device)
        y = torch.linspace(-1, 1, h, device=z.device)
        y_b, x_b = torch.meshgrid(y, x)
        # Expand from (h, w) -> (n, 1, h, w)
        x_b = x_b.expand(n, 1, -1, -1)
        y_b = y_b.expand(n, 1, -1, -1)
        # Concatenate along the channel dimension: final shape = (n, z_dim + 2, h, w)
        z_sb = torch.cat((z_b, x_b, y_b), dim=1)
        return z_sb

    def forward(self, x, log_m_k, background=False):
        """
        :param x: Input image
        :param log_m_k: Attention mask logits
        :return: x_k and reconstructed mask logits
        """
        params = self.encoder(torch.cat((x, log_m_k), dim=1))
        z_mu = params[:, :self._z_dim]
        z_logvar = params[:, self._z_dim:]
        z = self.reparameterize(z_mu, z_logvar)

        # "The height and width of the input to this CNN were both 8 larger than the target output (i.e. image) size
        #  to arrive at the target size (i.e. accommodating for the lack of padding)."
        h, w = x.shape[-2:]
        z_sb = self.spatial_broadcast(z, h + 8, w + 8)

        output = self.decoder(z_sb)
        x_mu = output[:, :self._input_nc]
        x_logvar = self._bg_logvar if background else self._fg_logvar
        m_logits = output[:, self._input_nc:]

        return m_logits, x_mu, x_logvar, z_mu, z_logvar


class AttentionBlock(nn.Module):

    def __init__(self, input_nc, output_nc, resize=True):
        super().__init__()
        self.conv = nn.Conv2d(input_nc, output_nc, 3, padding=1, bias=False)
        self.norm = nn.InstanceNorm2d(output_nc, affine=True)
        self._resize = resize

    def forward(self, *inputs):
        downsampling = len(inputs) == 1
        x = inputs[0] if downsampling else torch.cat(inputs, dim=1)
        x = self.conv(x)
        x = self.norm(x)
        x = skip = F.relu(x)
        if self._resize:
            x = F.interpolate(skip, scale_factor=0.5 if downsampling else 2., mode='nearest')
        return (x, skip) if downsampling else x


class Attention(nn.Module):
    """Create a Unet-based generator"""

    def __init__(self, input_nc, output_nc, ngf=32):
        """Construct a Unet generator
        Parameters:
            input_nc (int)  -- the number of channels in input images
            output_nc (int) -- the number of channels in output images
            ngf (int)       -- the number of filters in the last conv layer

        We construct the U-Net from the innermost layer to the outermost layer.
        It is a recursive process.
        """
        super(Attention, self).__init__()
        self.downblock1 = AttentionBlock(input_nc + 1, ngf)
        self.downblock2 = AttentionBlock(ngf, ngf * 2)
        self.downblock3 = AttentionBlock(ngf * 2, ngf * 4)
        self.downblock4 = AttentionBlock(ngf * 4, ngf * 8)
        self.downblock5 = AttentionBlock(ngf * 8, ngf * 8, resize=False)
        # no resizing occurs in the last block of each path
        # self.downblock6 = AttentionBlock(ngf * 8, ngf * 8, resize=False)

        self.mlp = nn.Sequential(
            nn.Linear(8 * 8 * ngf * 8, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 8 * 8 * ngf * 8),
            nn.ReLU(),
        )

        # self.upblock1 = AttentionBlock(2 * ngf * 8, ngf * 8)
        self.upblock2 = AttentionBlock(2 * ngf * 8, ngf * 8)
        self.upblock3 = AttentionBlock(2 * ngf * 8, ngf * 4)
        self.upblock4 = AttentionBlock(2 * ngf * 4, ngf * 2)
        self.upblock5 = AttentionBlock(2 * ngf * 2, ngf)
        # no resizing occurs in the last block of each path
        self.upblock6 = AttentionBlock(2 * ngf, ngf, resize=False)

        self.output = nn.Conv2d(ngf, output_nc, 1)

    def forward(self, x, log_s_k):
        tmpx = x
        # Downsampling blocks
        x, skip1 = self.downblock1(torch.cat((x, log_s_k), dim=1))
        x, skip2 = self.downblock2(x)
        x, skip3 = self.downblock3(x)
        x, skip4 = self.downblock4(x)
        x, skip5 = self.downblock5(x)
        skip6 = skip5
        # The input to the MLP is the last skip tensor collected from the downsampling path (after flattening)
        # _, skip6 = self.downblock6(x)
        # Flatten
        x = skip6.flatten(start_dim=1)
        x = self.mlp(x)
        # Reshape to match shape of last skip tensor
        x = x.view(skip6.shape)
        # Upsampling blocks
        # x = self.upblock1(x, skip6)
        x = self.upblock2(x, skip5)
        x = self.upblock3(x, skip4)
        x = self.upblock4(x, skip3)
        x = self.upblock5(x, skip2)
        x = self.upblock6(x, skip1)
        # Output layer
        logits = self.output(x)
        x = F.logsigmoid(logits)
        return x, logits
