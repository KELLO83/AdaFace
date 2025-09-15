import torch
from net import Backbone
import torchinfo
from torch.nn import Dropout
from torch.nn import Sequential
from torch.nn import Linear , Flatten
from torch.nn import BatchNorm1d, BatchNorm2d

Weight = 'adaface_ir101_webface12m.ckpt'
#Weight = 'adaface_ir101_webface4m.ckpt'
backbone = Backbone([224,224], 100, 'ir')


# Weight = 'adaface_ir50_casia.ckpt'
# backbone = Backbone([112,112],50,'ir')

print(backbone)

del backbone.output_layer

for name , param in backbone.named_parameters():
    print(f"name: {name}, requires_grad: {param.requires_grad}")
        

ckpt = torch.load(Weight, map_location='cpu')



load_result = backbone.load_state_dict({key.replace('model.', ''):val
                                        for key,val in ckpt['state_dict'].items() if 'model.' in key},
                                       strict=False)

print("누락된 가중치 : {}".format(load_result.missing_keys))
print("예상치못한 가중치 : {}".format(load_result.unexpected_keys))
print(load_result)
print("="*30)
input("아무키를 누르세요...")


backbone.output_layer =Sequential(BatchNorm2d(512),
                                        Dropout(0.4), Flatten(),
                                        Linear(512 * 14 * 14, 512),
                                        BatchNorm1d(512, affine=False))

dummy_input = torch.randn(1, 3, 112, 112)
model_info = torchinfo.summary(
    backbone,
    input_size=(1, 3, 224, 224),
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

