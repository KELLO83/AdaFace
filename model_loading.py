import torch
from net import Backbone
import torchinfo

Weight = 'adaface_ir101_webface12m.ckpt'
Weight = 'adaface_ir101_webface4m.ckpt'
backbone = Backbone([112,112], 100, 'ir')


# Weight = 'adaface_ir50_casia.ckpt'
# backbone = Backbone([112,112],50,'ir')

ckpt = torch.load(Weight, map_location='cpu')


load_result = backbone.load_state_dict({key.replace('model.', ''):val
                                        for key,val in ckpt['state_dict'].items() if 'model.' in key})

print("누락된 가중치 : {}".format(load_result.missing_keys))
print("예상치못한 가중치 : {}".format(load_result.unexpected_keys))
print(load_result)
print("="*30)



dummy_input = torch.randn(1, 3, 112, 112)
model_info = torchinfo.summary(
    backbone,
    input_size=(100, 3, 112, 112),
    verbose=False,
    col_names=["input_size", "output_size", "num_params", "trainable","params_percent"],
    row_settings=["depth"],
    device='cuda:1',
    mode='eval'
)
print(model_info)
# backbone.eval()
# backbone = backbone.to('cuda:1')
# dummy_input = dummy_input.to(torch.float32).to('cuda:1')
# output  = backbone(dummy_input)
