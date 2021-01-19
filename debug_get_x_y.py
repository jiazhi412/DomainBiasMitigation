import utils
import os
import h5py
from models import dataloader
import torchvision.transforms as transforms
import torch

save_path = 'record/celeba_baseline/debug/'
test_result = utils.load_pkl(os.path.join(save_path, 'test_result.pkl'))
feature = test_result['feature']

data_setting = {
            'image_feature_path': './data/celeba/celeba.h5py',
            'target_dict_path': './data/celeba/labels_dict',
            'train_key_list_path': './data/celeba/train_key_list',
            'dev_key_list_path': './data/celeba/dev_key_list',
            'test_key_list_path': './data/celeba/test_key_list',
            'subclass_idx_path': './data/celeba/subclass_idx',
            'augment': True
        }
image_feature = h5py.File(data_setting['image_feature_path'], 'r')
target_dict = utils.load_pkl(data_setting['target_dict_path'])
train_key_list = utils.load_pkl(data_setting['train_key_list_path'])
dev_key_list = utils.load_pkl(data_setting['dev_key_list_path'])
test_key_list = utils.load_pkl(data_setting['test_key_list_path'])

targetss = []
for key in test_key_list:
    targetss.append(target_dict[key])
targetss = targetss[:100]
targetss = torch.tensor(targetss)

label = targetss[:,:38]
gender = targetss[:,39]

# normalize according to ImageNet
mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]
normalize = transforms.Normalize(mean=mean, std=std)

transform_test = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            normalize,
        ])


test_data = dataloader.CelebaDataset(test_key_list, image_feature,
                                     target_dict, transform_test)
test_loader = torch.utils.data.DataLoader(
    test_data, batch_size = 100,
    shuffle=False, num_workers=1)
#
# self.test_target = np.array([target_dict[key] for key in test_key_list])
# self.test_class_weight = utils.compute_class_weight(self.test_target)

# test_targets = None
# for images, targets in test_loader:
#     test_targets = targets

images, targets = next(iter(test_loader))
del images
# print(targets)
print(image_feature.size())
print(targets.size())
print(targets == targetss)
a = 10