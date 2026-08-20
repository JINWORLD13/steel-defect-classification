"""
Grad-CAM 직접 구현 — "모델이 어디를 보고 판정했나"를 히트맵으로

원리 세 줄:
  1) 마지막 합성곱 층(layer4)의 출력 A_k는 512장의 7x7 '특징 지도'임
  2) 정답 클래스 점수를 A_k로 미분한 기울기의 평균 α_k = "이 지도가 판정에 얼마나 중요한가"
  3) Σ α_k·A_k 에 ReLU를 씌우면 '판정에 기여한 위치'만 남음 — 그게 히트맵

라이브러리를 안 쓰고 직접 짜는 이유: hook(중간 계산 가로채기)과 backward를
이해해야 하고, 아래 자기검증(CAM 항등식)으로 구현이 맞다는 걸 증명할 수 있어서임.

주의: @torch.no_grad()를 쓰면 안 됨 — 기울기가 곧 재료라서, 끄면 hook에 아무것도 안 잡힘.
"""
import torch
import torch.nn.functional as F


class GradCAM:
    """model의 layer4에 훅을 걸어 히트맵을 뽑는 도구."""

    def __init__(self, model):
        self.model = model.eval()
        self.act = None                       # 순전파 때 잡은 특징 지도 A (1,512,7,7)
        self.grad = None                      # 역전파 때 잡은 기울기 dY/dA (1,512,7,7)
        # forward hook: layer4가 출력을 낼 때 그 값을 옆으로 복사해 둠
        model.layer4.register_forward_hook(self._save_act)
        # backward hook: 역전파가 layer4를 지날 때 기울기를 복사해 둠
        model.layer4.register_full_backward_hook(self._save_grad)

    def _save_act(self, module, inp, out):
        self.act = out.detach()

    def _save_grad(self, module, grad_in, grad_out):
        self.grad = grad_out[0].detach()

    def __call__(self, img, class_idx=None):
        """img: 정규화된 (3,224,224) 텐서 -> (7,7) 히트맵 (0~1 정규화, 전멸이면 전부 0)"""
        logits = self.model(img.unsqueeze(0))          # no_grad 금지 — 기울기가 재료임
        if class_idx is None:
            class_idx = int(logits.argmax(1))
        self.model.zero_grad()
        logits[0, class_idx].backward()                 # 그 클래스 점수만 역전파

        alpha = self.grad[0].mean(dim=(1, 2))           # (512,) 지도별 중요도
        cam = F.relu((alpha[:, None, None] * self.act[0]).sum(0))  # (7,7) 가중합 + ReLU
        if cam.max() > 0:
            cam = cam / cam.max()                       # 0~1로 정규화
        return cam, class_idx


def upsample(cam, size=224):
    """(7,7) -> (224,224). 픽셀 좌표에서 박스와 겹침을 재기 위함."""
    return F.interpolate(cam[None, None], size=(size, size),
                         mode="bilinear", align_corners=False)[0, 0]


def self_check(model):
    """구현 자기검증 — Grad-CAM ≡ CAM 항등식.

    ResNet18의 끝은 GAP(공간 평균) -> fc 라서, 수학적으로
    Grad-CAM의 α_k 가 정확히 fc 가중치 W[c,k]/49 가 됨.
    즉 fc 가중치로 직접 만든 CAM과 (정규화 후) 완전히 같아야 함.
    다르면 hook 어딘가가 틀린 것.
    """
    from common import DEVICE
    torch.manual_seed(0)
    img = torch.randn(3, 224, 224).to(DEVICE)

    cam_grad, c = GradCAM(model)(img)

    with torch.no_grad():                              # CAM 쪽은 미분이 필요 없음
        feats = model.conv1(img.unsqueeze(0))
        feats = model.maxpool(model.relu(model.bn1(feats)))
        for layer in (model.layer1, model.layer2, model.layer3, model.layer4):
            feats = layer(feats)                       # (1,512,7,7)
        w = model.fc.weight[c]                         # (512,) 그 클래스의 fc 가중치
        cam_direct = F.relu((w[:, None, None] * feats[0]).sum(0))
        if cam_direct.max() > 0:
            cam_direct = cam_direct / cam_direct.max()

    diff = float((cam_grad - cam_direct).abs().max())
    print(f"자기검증: Grad-CAM ≡ CAM (최대차 {diff:.2e})",
          "통과" if diff < 1e-5 else "실패 — 구현 점검 필요")
    return diff < 1e-5


if __name__ == "__main__":
    from common import DEVICE, build_model
    m = build_model()
    m.load_state_dict(torch.load("best_model_v1.pth", map_location=DEVICE))
    assert self_check(m)
