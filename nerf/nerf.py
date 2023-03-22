import torch
import torch.nn as nn
import torch.nn.functional as F

class NerfModel(nn.Module):
    """the MLP for NeRF""" 
    def __init__(self, D=8, W=256, input_ch=3, input_ch_views=3, output_ch=4, skips=[4],use_viewdirs=False):
        super(NerfModel, self).__init__()
        self.D = D
        self.W = W
        self.input_ch = input_ch
        self.input_ch_views = input_ch_views
        self.skips = skips
        self.use_viewdirs = use_viewdirs

        self.pts_linears = nn.ModuleList(
            [nn.Linear(input_ch, W)] + [nn.Linear(W,W) if i not in self.skips else nn.Linear(W + input_ch, W) for i in range (D-1)]
        )
        self.views_linears = nn.ModuleList(
            [nn.Linear(input_ch_views + W, W//2)]
        )

        if use_viewdirs:
            self.feature_linear = nn.Linear(W, W)
            self.alpha_linear = nn.Linear(W, 1)
            self.rgb_linear = nn.Linear(W//2, 3)

        else:
            self.output_linear = nn.Linear(W, output_ch)
        # Sue
        # self.mlp = self.build_network() 

    def build_network(self):
        # Sue
        return None 
    
    def forward(self, x):
        # Sue
        '''
        @input:
        x: [nb,n,6]
        '''
        input_pts, input_views = torch.split(x, [self.input_ch, self.input_ch_views], dim = -1)
        h = input_pts
        for i, l in enumerate(self.pts_linears):
            h = self.pts_linears[i](h)
            h = F.relu(h)
            if i in self.skips:
                h = torch.cat([input_pts, h], -1)

        if self.use_viewdirs:
            alpha = self.alpha_linear(h)
            feature = self.feature_linear(h)
            h = torch.cat([feature, input_views], -1)
            for i, l in enumerate(self.views_linears):
                h = self.views_linears[i](h)
                h = F.relu(h)
            rgb = self.rgb_linear(h)
            outputs = torch.cat([rgb, alpha], -1)
        else:
            outputs = self.output_linear(h)

        return outputs

# def create_nerf(args):
#     model = NerfModel(D=args.netdepth, W=args.netwidth, input_ch=input_ch, output_ch=output_ch, skips=skips,input_ch_views=input_ch_views, use_viewdirs=args.use_viewdirs).to(device)
#     grad_vars = list(model.parameters())
#
#     model_fine = None
#     if args.N_importance > 0:
#         model_fine = NerfModel(D=args.netdepth_fine, W=args.netwidth_fine, input_ch=input_ch, output_ch=output_ch, skips=skips, input_ch_views=input_ch_views, use_viewdirs=args.use_viewdirs).to(device)
#         grad_vars += list(model_fine.parameters())