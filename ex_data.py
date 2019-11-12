from dataset import NYUDataset
from custom_transforms import *
from plot_utils import *
import model_utils

bs = 8
sz = (320,240)
mean, std = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
mean, std = torch.tensor(mean), torch.tensor(std)
unnormalize = UnNormalizeImgBatch(mean, std)

tfms = transforms.Compose([
    ResizeImgAndDepth(sz),
    RandomHorizontalFlip(),
    ImgAndDepthToTensor(),
    NormalizeImg(mean, std)
])
ds = NYUDataset('/content/gdrive/My Drive/data/', tfms).__getitem__(1)
# ds = NYUDataset('../data/', tfms).__getitem__(1)
plot_image(model_utils.get_unnormalized_ds_item(unnormalize,ds))
# plot_image(ds)