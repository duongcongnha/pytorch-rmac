import numpy as np
import torch
import torch.nn as nn





def rmac_regions(W, H, L): # input is size feature map
    '''
    return (x,y,w,h) of boxes/regions from a given size
    '''

    ovr = 0.4 # desired overlap of neighboring regions
    steps = np.array([2, 3, 4, 5, 6, 7], dtype=float) # possible regions for the long dimension

    w = min(W,H)

    b = (max(H,W) - w)/(steps-1)
    idx = np.argmin(abs(((w ** 2 - w*b)/w ** 2)-ovr)) # steps(idx) regions for long dimension

    # rregion overplus per dimension
    Wd, Hd = 0, 0
    if H < W:
        Wd = idx + 1
    elif H > W:
        Hd = idx + 1

    regions = []

    for l in range(1,L+1):

        wl = np.floor(2*w/(l+1))
        wl2 = np.floor(wl/2 - 1)

        b = (W - wl) / (l + Wd - 1)
        if np.isnan(b): # for the first level
            b = 0
        cenW = np.floor(wl2 + np.arange(0,l+Wd)*b) - wl2 # center coordinates

        b = (H-wl)/(l+Hd-1)
        if np.isnan(b): # for the first level
            b = 0
        cenH = np.floor(wl2 + np.arange(0,l+Hd)*b) - wl2 # center coordinates

        for i_ in cenH:
            for j_ in cenW:
                # R = np.array([i_, j_, wl, wl], dtype=int)
                R = np.array([j_, i_, wl, wl], dtype=int)
                if not min(R[2:]):
                    continue

                regions.append(R)

    regions = np.asarray(regions)
    return regions



class RoiPooling(nn.Module):

    """
    Roi pooling for chanel last
    input shape: (batch_size, h, w, num_chanels)
    output shape: (batch_size, num_rois * sum([i * i for i in pool_list]), num_chanels)
    """

    def __init__(self, pool_list, num_rois) -> None:
        super().__init__()
        self.pool_list = pool_list
        self.num_rois = num_rois

    def forward(self, feature_map, rois):
        
        outputs = []
        for roi_idx in range(self.num_rois):

            x = rois[roi_idx][0]
            y = rois[roi_idx][1]
            w = rois[roi_idx][2]
            h = rois[roi_idx][3]

            row_length = [w / i for i in self.pool_list]
            col_length = [h / i for i in self.pool_list]

            for pool_num, num_pool_regions in enumerate(self.pool_list):
                for ix in range(num_pool_regions):
                    for jy in range(num_pool_regions):
                        x1 = x + ix * col_length[pool_num]
                        x2 = x1 + col_length[pool_num]
                        y1 = y + jy * row_length[pool_num]
                        y2 = y1 + row_length[pool_num]

                        x1 = torch.round(x1).type(torch.int32)
                        x2 = torch.round(x2).type(torch.int32)
                        y1 = torch.round(y1).type(torch.int32)
                        y2 = torch.round(y2).type(torch.int32)

                        x_crop = feature_map[:, y1:y2, x1:x2, :]
                        pooled_val = torch.amax(x_crop, dim=(1, 2))
                        outputs.append(pooled_val)
        final_output = torch.stack(outputs)
        final_output = torch.permute(final_output, (1,0,2))

        return final_output




class RMAC(nn.Module):
    def __init__(self, pca_weight_path:str, h=14, w=14) -> None:
        super().__init__()
        self.pca_layer = torch.nn.Linear(512, 512, bias=True)
        self.pca_layer.load_state_dict(torch.load(pca_weight_path))

        self.regions = rmac_regions(h, w, 3)
        num_rois = len(self.regions)
        self.regions = torch.from_numpy(self.regions).type(torch.float64)
        
        ROI_output_sizes = [1, 2]
        self.roi_pooling = RoiPooling(ROI_output_sizes, num_rois)
        

    def forward(self, x):
        '''
        Do RMAC for batch of feature maps
            x: (batch_size, H, W, num_channels)
        '''
        x = self.roi_pooling(x, self.regions)
        x = torch.nn.functional.normalize(x, p=2, dim=2)
        x = self.pca_layer(x)
        x = torch.nn.functional.normalize(x, p=2, dim=2)
        x = torch.sum(x, dim=1)
        x = torch.nn.functional.normalize(x, p=2, dim=1)

        return x

if __name__=='__main__':
    rmac_model = RMAC(pca_weight_path='pca_weight_torch.pth') # my weight is just used for 512 chanels -> 512 chanels. To get weights to fit your data, you should train a siamese network on your custom dataset

    input = torch.rand(8, 14, 14, 512)
    output = rmac_model(input)
    print(output.shape)