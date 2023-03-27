import torch
import torch.nn as nn
import torch.nn.functional as F

class PositionalEncoder():
    def __init__(self, n_freq):
        self.n_freq = n_freq
        self.freq_bands = 2. ** torch.linspace(0., n_freq - 1, n_freq)
        self.embed_fns = []
        for freq in self.freq_bands:
            for p_fn in [torch.sin, torch.cos]:
                self.embed_fns.append(lambda x, p_fn=p_fn, freq=freq : p_fn(x * freq))

    def encode(self, x):
        encoded = torch.cat([fn(x) for fn in self.embed_fns], -1)
        return encoded, encoded.shape[-1]
    
class NerfModel(nn.Module):
    """the MLP for NeRF""" 
    def __init__(self, D=8, W=256, input_ch=3, input_ch_views=3, output_ch=4, skips=[4], use_viewdirs=False):
        super(NerfModel, self).__init__()
        self.D = D
        self.W = W
        self.skips = skips
        self.use_viewdirs = use_viewdirs
        self.pos_encoder = PositionalEncoder(n_freq = 10)
        self.views_encoder = PositionalEncoder(n_freq = 4)
        self.input_ch = input_ch * (self.pos_encoder.n_freq*2)
        self.input_ch_views = input_ch_views * (self.views_encoder.n_freq*2)

        self.pts_linears = nn.ModuleList(
            [nn.Linear(self.input_ch, W)] + [nn.Linear(W, W) if i not in self.skips else nn.Linear(W + self.input_ch, W) for i in range (D-1)]
        )
        self.views_linears = nn.ModuleList(
            [nn.Linear(self.input_ch_views + W, W//2)]
        )

        if use_viewdirs:
            self.feature_linear = nn.Linear(W, W)
            self.alpha_linear = nn.Linear(W, 1)
            self.rgb_linear = nn.Linear(W//2, 3)
        else:
            self.output_linear = nn.Linear(W, output_ch)

    def encode_input(self, x):
        '''
        @input:
        x: [n, 6]
        '''
        pts = x[:,:3]
        views = x[:,3:]
        pts_encoded, pts_encoded_ch = self.pos_encoder.encode(pts)
        views_encoded, views_encoded_ch = self.views_encoder.encode(views)
        self.input_ch = pts_encoded_ch
        self.input_ch_views = views_encoded_ch
        return pts_encoded, views_encoded
    
    def forward(self, x):
        # Sue
        '''
        @input:
        '''
        input_pts, input_views = torch.split(x, [self.input_ch, self.input_ch_views], dim = -1)
        h = input_pts
        for i, l in enumerate(self.pts_linears):
            h = self.pts_linears[i](h)
            # if torch.isnan(h).any() or torch.isinf(h).any():
            #     print(i)
            #     print("h is nan bf relu")
            #     print(h)
            #     exit()
            h = F.relu(h)
            if i in self.skips:
                h = torch.cat([input_pts, h], -1)
            # if torch.isnan(h).any() or torch.isinf(h).any():
            #     print(i)
            #     print("h is nan")
            #     print(h)
            #     exit()

        if self.use_viewdirs:
            alpha = self.alpha_linear(h)
            feature = self.feature_linear(h)
            h = torch.cat([feature, input_views], -1)
            for i, l in enumerate(self.views_linears):
                h = self.views_linears[i](h)
                h = F.relu(h)
            if torch.isnan(h).any() or torch.isinf(h).any():
                print("h view is nan")
                print(h)
                exit()
            rgb = self.rgb_linear(h)
            outputs = torch.cat([rgb, alpha], -1)
        else:
            outputs = self.output_linear(h)

        if torch.isnan(outputs).any() or torch.isinf(outputs).any():
            print("output is nan")
            print(outputs)
            exit()
        return outputs

# def create_nerf(args):
#     model = NerfModel(D=args.netdepth, W=args.netwidth, input_ch=input_ch, output_ch=output_ch, skips=skips,input_ch_views=input_ch_views, use_viewdirs=args.use_viewdirs).to(device)
#     grad_vars = list(model.parameters())
#
#     model_fine = None
#     if args.N_importance > 0:
#         model_fine = NerfModel(D=args.netdepth_fine, W=args.netwidth_fine, input_ch=input_ch, output_ch=output_ch, skips=skips, input_ch_views=input_ch_views, use_viewdirs=args.use_viewdirs).to(device)
#         grad_vars += list(model_fine.parameters())