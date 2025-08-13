from net import Backbone
import torch


ckpt = torch.load('adaface_ir50_webface4m.ckpt', map_location='cpu')
print("체크포인트 키들:", list(ckpt.keys()))

backbone = Backbone(input_size=(112, 112, 3), num_layers=50)

filtered_state_dict = {key.replace('model.', ''): val 
                      for key, val in ckpt['state_dict'].items() if 'model.' in key}

print("필터링된 키 개수:", len(filtered_state_dict))
print("키:", list(filtered_state_dict.keys())[:50])


load_result = backbone.load_state_dict(filtered_state_dict, strict=False)

print("누락된 가중치:", load_result.missing_keys)
print("예상치 못한 가중치:", load_result.unexpected_keys)

print("="*30)