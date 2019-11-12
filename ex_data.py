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
# ds = NYUDataset('/content/gdrive/My Drive/data/', tfms).__getitem__(1)
ds = NYUDataset('../data/', tfms)

print(f'{ds[0][0].shape} !! {ds[0][1].shape}')
print(f'{ds[1][0].shape} !! {ds[1][1].shape}')
plt.imshow(ds[0][0].cpu().numpy().transpose((1, 2, 0)))
plt.show()

plt.imshow(ds[0][1])
plt.show()


# ds = NYUDataset('../data/', tfms).__getitem__(1)
# print(f'{ds[1].shape}')
# # plt.imshow(ds[1].cpu().numpy().transpose((1,2,0)))
# plt.imshow(ds[1].cpu().numpy())
# plt.show()