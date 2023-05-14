# import pydevd_pycharm
#
# pydevd_pycharm.settrace('10.160.144.172', port=9996, stdoutToServer=True, stderrToServer=True)
import torch
from collections import OrderedDict

mode_dict = OrderedDict()
# model_file_path = "work_dirs/leformer_256x256_qtpl_160k/iter_144000.pth"
# model_file_path = "work_dirs/segformer_mit-b0_256x256_160k_qtpl/iter_160000.pth"
# model_file_path = "work_dirs/fpn_poolformer_s12_8x4_256x256_160k_qtpl/iter_160000.pth"
model_file_path = "work_dirs/fcn_unet_s5-d16_4x4_256x256_160k_qtpl/iter_160000.pth"

save_file_path = "/gpfs/home/chenben/models/pre_train"
model = torch.load(model_file_path)
for item in model['state_dict']:
    # print(item)
    list = item.split('.')[1:]
    temp = '.'.join([i for i in list])
    # print(temp)
    mode_dict[temp] = model['state_dict'][item]

model['state_dict'] = mode_dict
torch.save(model, f"{save_file_path}/fcn_unet_qtpl_160k.pth")
# torch.save(model, f"{save_file_path}/surface_water_128k.pth")

# for item in model:
#     list = item.split('.')
#     if (list[0] == 'layers'):
#         list[0] = 'transformer_encoder_layers'
#     # print(list)
#     temp = '.'.join([i for i in list])
#     # print(temp)
#     # item = temp
#     # print(item)
#     mode_dict[temp] = model[item]
#
# torch.save(mode_dict, "/gpfs/home/chenben/models/LEFormer/leformer.pth")

