import numpy as np
import torch


class Conv(torch.nn.Module):
    def __init__(self, dim_in, dim_out):
        super().__init__()
        dim_middle = 10*dim_in
        kernel_size = 3
        depth = 3
        padding = 'same'
        layers = []
        if depth == 1:
            layers.append(torch.nn.Conv2d(dim_in, dim_out, kernel_size=kernel_size, padding=padding))
        else:
            for i in range(depth-2):
                if i == 0:
                    layers.append(torch.nn.Conv2d(dim_in, dim_middle, kernel_size=kernel_size, padding=padding))
                else:
                    layers.append(torch.nn.Conv2d(dim_middle, dim_middle, kernel_size=kernel_size, padding=padding))

                layers.append(torch.nn.ReLU(inplace=True))

            layers.append(torch.nn.Conv2d(dim_middle, dim_out, kernel_size=kernel_size, padding=padding))

        #layers.append(torch.nn.ReLU(inplace=True))
        layers.append(torch.nn.Sigmoid())
        self.network = torch.nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)


class MLP(torch.nn.Module):
    def __init__(self, dim_in, dim_out):
        super().__init__()
        dim_middle = 10*dim_in
        depth = 3
        if depth==1:
            layers = [torch.nn.Linear(dim_in, dim_out),  # (H, W, dim_in) -> (H, W, dim_out)
                      torch.nn.Sigmoid()  # torch.nn.Softmax(dim=0)  # to make sure the output is in [0,1]
            ]
        else:
            layers = []
            for i in range(depth-2):
                if i == 0:
                    layers.append(torch.nn.Linear(dim_in, dim_middle))
                else:
                    layers.append(torch.nn.Linear(dim_middle, dim_middle))

                layers.append(torch.nn.ReLU(inplace=True))

            layers.append(torch.nn.Linear(dim_middle, dim_out))
            layers.append(torch.nn.Sigmoid())  # to make sure the output is in [0,1] .Softmax(dim=0)

        self.network = torch.nn.Sequential(*layers)

    def forward(self, x):
        res = self.network(x.permute(1, 2, 0)).permute(2, 0, 1)
        return res


def compact(p, num_bands, num_channels):
    assert p.dim() == 3
    assert p.shape[0] == num_bands * num_channels
    # p is bands * channels, H, W
    x = []
    for k in range(num_channels):
        x.append(p[k * num_bands: (k + 1) * num_bands].sum(axis=0))
    res = torch.stack(x, 0)
    assert res.shape[0] == num_channels
    return res


def myloss(pred, gt, distance='l1'):
    assert distance in ['l1', 'l2']
    assert gt.dim() == 3
    assert pred.dim() == 3
    assert gt.shape == pred.shape
    assert (0 <= gt.max() <= 1.0) and (0 <= pred.max() <= 1.0)

    #print('gt shape, min, max {} {} {} pred shape, min max {} {} {}'
    #      .format(gt.shape, gt.min(), gt.max(), pred.shape, pred.min(), pred.max()))

    diff = pred - gt  # 255*(pred - gt)
    # print('diff mean {}'.format(diff.mean()))

    if distance == 'l1':
        return torch.abs(diff).mean()
    elif distance == 'l2':
        return (diff ** 2).mean()
    else:
        assert False



if __name__ == "__main__":
    import os

    path_images = './data/multi-modal-studio/birdhouse/images'
    # undistorted images, all channels into a numpy array of shape C, H, W
    fname = os.listdir(path_images)[0]
    dim_embedding = 16 # 32 # 16
    # To be changed in:
    # submodules/diff-gaussian-rasterization/cuda_rasterizer/config.h
    # arguments/__init__.py
    # scene/gaussian_model.py
    # then hyper-gaussian-grouping-tfg/submodules/diff-gaussian-rasterization$ pip install .
    Model = Conv # MLP # Conv

    # gt = torch.tensor(imread(os.path.join(path_images, fname)) / 2**16) # H, W, 3 (RGB) in the interval [0,1]
    gt = np.load(os.path.join(path_images, fname))
    gt = torch.tensor(gt)
    print('gt {}, min {}, max {}'.format(gt.shape, gt.min(), gt.max()))
    # plt.figure(), plt.imshow(gt.permute(1,2,0)), plt.show(block=False)
    channels, height, width = gt.shape

    embedding = torch.rand((dim_embedding, height, width))
    print('embedding {}, min {}, max {}'.format(embedding.shape, embedding.min(), embedding.max()))

    model = Model(dim_embedding, channels, depth=1)
    pred = model(embedding) # C, H, W
    print('pred {}, min {}, max {}'.format(pred.shape, pred.min(), pred.max()))

    l1_loss_2 = myloss(pred, gt, 'l1')
    l2_loss_2 = myloss(pred, gt, 'l2')
    print()
    print('l1 loss {}'.format(l1_loss_2))
    print('l2 loss {}'.format(l2_loss_2))

