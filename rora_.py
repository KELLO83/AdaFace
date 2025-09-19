import torch
import net
from convrora_bn import build_backbone, merge_lora_modules

ckpt_path = "best_adabn_lora.ckpt"          # 기존 LoRA 포함 체크포인트
save_path = "best_adabn_lora_merged.ckpt"   # 병합 후 저장 경로
arch = "ir_101"                             # 사용한 백본 아키텍처(필요 시 수정)

# 백본 로드
backbone = build_backbone(arch, 112)
checkpoint = torch.load(ckpt_path, map_location="cpu")

# LoRA 가중치 로드 (state_dict 형태라고 가정)
if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
    state = {k.replace("model.", ""): v for k, v in checkpoint["state_dict"].items()}
    backbone.load_state_dict(state, strict=False)
elif isinstance(checkpoint, dict) and "backbone" in checkpoint:
    backbone.load_state_dict(checkpoint["backbone"], strict=False)
else:
    backbone.load_state_dict(checkpoint, strict=False)

# 로라 병합 -> 순수 Conv 가중치로 변환
merge_lora_modules(backbone)

# 저장
merged_state = backbone.state_dict()
if isinstance(checkpoint, dict):
    if "backbone" in checkpoint:
        checkpoint["backbone"] = merged_state
    elif "state_dict" in checkpoint:
        for key in list(checkpoint["state_dict"].keys()):
            trimmed = key.replace("model.", "")
            if trimmed in merged_state:
                checkpoint["state_dict"][key] = merged_state[trimmed]
else:
    checkpoint = merged_state

torch.save(checkpoint, save_path)
print(f"Saved merged checkpoint to {save_path}")
