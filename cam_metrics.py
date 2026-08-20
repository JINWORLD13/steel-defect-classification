"""
히트맵 채점 — "박스 라벨이 있으니 설명도 숫자로 채점할 수 있다"

지표 3종 (전부 정답 박스 마스크와의 겹침):
  PG (pointing game) : 히트맵의 최고점이 박스 안에 있는가 (맞다/아니다)
  EBPG               : 히트맵 에너지(값의 합) 중 박스 안 비율 (0~1)
  ADR                : 박스 안 평균 밀도 / 전체 평균 밀도 — 1.0이면 무작위와 같음

★ ADR의 함정을 같이 보고함:
  ADR의 이론적 상한은 1/(박스 면적 비율)이라 클래스마다 다름.
  박스가 화면을 덮는 클래스(pitted_surface)는 아무리 잘해도 ADR이 낮게 나옴.
  그래서 상한을 반드시 병기하고, 클래스 간 비교엔 정규화판
  ADR* = (ADR-1)/(상한-1) 을 씀 (1=이론적 완벽, 0=무작위).

실행: ./venv/bin/python cam_metrics.py
"""
import torch
from PIL import Image

from common import CLASSES, DEVICE, build_model, save_result
from corruptions import clean_transform, make_corrupt_transform
from gradcam import GradCAM, upsample
from labels import boxes_of, read_manifest, stem_of


def mask_of(stem, cls_idx, size=224):
    """그 클래스의 박스들을 합친 (224,224) 참/거짓 마스크."""
    m = torch.zeros(size, size, dtype=torch.bool)
    for c, cx, cy, w, h in boxes_of(stem):
        if c == cls_idx:
            x1, y1 = int((cx - w / 2) * size), int((cy - h / 2) * size)
            x2, y2 = int((cx + w / 2) * size), int((cy + h / 2) * size)
            m[max(0, y1):min(size, y2), max(0, x1):min(size, x2)] = True
    return m


def score_cam(cam224, mask):
    """히트맵 하나를 세 지표로 채점. 전멸(전부 0)이면 None을 돌려줌."""
    total = float(cam224.sum())
    if total == 0:
        return None                                     # dead CAM — 따로 셈
    area = float(mask.float().mean())                   # 박스 면적 비율
    inside = float(cam224[mask].sum())
    y, x = divmod(int(cam224.argmax()), cam224.shape[1])
    return {
        "pg": bool(mask[y, x]),                         # 최고점이 박스 안?
        "ebpg": inside / total,                         # 에너지 비율
        "adr": (inside / max(area, 1e-9)) / total,      # 밀도비 = EBPG/면적
        "cap": 1.0 / max(area, 1e-9),                   # ADR의 이론적 상한
    }


def evaluate(model, transform, target_from_filename=True):
    """test 270장을 채점해 클래스별로 묶음."""
    cam_tool = GradCAM(model)
    per_cls = {c: [] for c in CLASSES}
    dead = 0
    for path in read_manifest("test"):
        stem = stem_of(path)
        cls_idx = CLASSES.index(stem.rsplit("_", 1)[0])
        img = transform(Image.open(path).convert("RGB")).to(DEVICE)
        cam, _ = cam_tool(img, class_idx=cls_idx)       # 정답 클래스 기준의 설명
        s = score_cam(upsample(cam).cpu(), mask_of(stem, cls_idx))
        if s is None:
            dead += 1
        else:
            per_cls[CLASSES[cls_idx]].append(s)
    return per_cls, dead


def summarize(per_cls, dead, name):
    n_total = sum(len(v) for v in per_cls.values()) + dead
    rows = {}
    print(f"\n===== {name} (test {n_total}장, 전멸 {dead}장 제외 채점) =====")
    print(f"{'class':18s} {'PG':>6s} {'EBPG':>6s} {'ADR':>6s} {'상한':>6s} {'ADR*':>6s}")
    for c in CLASSES:
        v = per_cls[c]
        if not v:
            continue
        pg = sum(s["pg"] for s in v) / len(v)
        ebpg = sum(s["ebpg"] for s in v) / len(v)
        adr = sum(s["adr"] for s in v) / len(v)
        cap = sum(s["cap"] for s in v) / len(v)
        star = (adr - 1) / (cap - 1) if cap > 1 else 0.0    # 정규화판 (0=무작위, 1=완벽)
        print(f"{c:18s} {pg:6.3f} {ebpg:6.3f} {adr:6.2f} {cap:6.2f} {star:6.3f}")
        rows[c] = {"pg": round(pg, 4), "ebpg": round(ebpg, 4), "adr": round(adr, 3),
                   "cap": round(cap, 2), "adr_star": round(star, 4), "n": len(v)}
    allv = [s for v in per_cls.values() for s in v]
    macro = {
        "pg": round(sum(s["pg"] for s in allv) / len(allv), 4),
        "ebpg": round(sum(s["ebpg"] for s in allv) / len(allv), 4),
        "adr": round(sum(s["adr"] for s in allv) / len(allv), 3),
        "dead": dead, "dead_rate": round(dead / n_total, 4),
    }
    print(f"{'전체(live만)':16s} {macro['pg']:6.3f} {macro['ebpg']:6.3f} {macro['adr']:6.2f}"
          f"   | 전멸률 {macro['dead_rate']:.1%}")
    return {"per_class": rows, "macro": macro}


def main():
    model = build_model()
    model.load_state_dict(torch.load("best_model_v1.pth", map_location=DEVICE))

    result = {}
    per_cls, dead = evaluate(model, clean_transform())
    result["clean"] = summarize(per_cls, dead, "v1 · 깨끗한 test")

    # 열화가 히트맵을 어떻게 무너뜨리나 — '살아남은 것의 질'과 '전멸률'을 분리해 봄
    for kind in ("gaussian_noise", "gaussian_blur", "low_contrast"):
        per_cls, dead = evaluate(model, make_corrupt_transform(kind, 3))
        result[f"{kind}_s3"] = summarize(per_cls, dead, f"v1 · {kind} s3")

    save_result("cam", result)


if __name__ == "__main__":
    main()
