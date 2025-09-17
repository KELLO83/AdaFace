from net import Backbone
import torch



Weight = 'adaface_ir101_webface12m.ckpt'
backbone = Backbone([112,112], 100, 'ir')
ckpt = torch.load(Weight, map_location='cpu')
load_result = backbone.load_state_dict({key.replace('model.', ''):val
                                        for key,val in ckpt['state_dict'].items() if 'model.' in key},
                                       strict=False)

for name , param in backbone.named_parameters():
    param.requires_grad = False

for name , param in backbone.named_parameters():
    if name.startswith('body.48') or name.startswith('body.47') or name.startswith('body.46') or name.startswith('body.45') or name.startswith('body.44'):
        param.requires_grad = True

# for name , param in backbone.named_parameters():
#     print(f"name: {name}, requires_grad: {param.requires_grad}")
        

print("Verifying parameter freeze status:")
total_params = 0
trainable_params = 0
for name, param in backbone.named_parameters():
    total_params += param.numel()
    if param.requires_grad:
        trainable_params += param.numel()
        print(f"- {name} (Trainable)")

print(f"Total parameters: {total_params}")
print(f"Trainable parameters: {trainable_params}")
print(f"Percentage of trainable parameters: {100.0 * trainable_params / total_params:.2f}%")


print("누락된 가중치 : {}".format(load_result.missing_keys))
print("예상치못한 가중치 : {}".format(load_result.unexpected_keys))
print(load_result)
print("="*30)

for name , param in backbone.named_parameters():
    print(f"name: {name}, requires_grad: {param.requires_grad}")
