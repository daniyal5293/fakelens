"""FakeLens Gradio demo.

    python app/app.py --clip checkpoints/A2_head.pt --resnet checkpoints/B_model.pt
    (or set FAKELENS_CLIP_CKPT / FAKELENS_RESNET_CKPT; runs on CPU)

The output is a *model prediction*, not proof of where an image came from.
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from fakelens.analysis.explain import gradcam_resnet, occlusion_map, overlay  # noqa: E402
from fakelens.data.transforms import denormalize, make_perturbation  # noqa: E402
from fakelens.detector import Detector  # noqa: E402

DISCLAIMER = (
    "**FakeLens is an academic demo.** It outputs a *model prediction*, not forensic proof. "
    "It was trained on Stable Diffusion v1.4 images (GenImage) and can be wrong, especially on "
    "generators, image types or edits it has not seen. Do not use it for any high-stakes decision. "
    "Non-commercial use only (training data licence: CC BY-NC-SA 4.0)."
)


def build_app(dets: dict):
    import gradio as gr

    def run(img, jpeg_q, scale, explain):
        if img is None:
            return {}, None, ""
        img = img.convert("RGB")
        if scale != 1.0:
            img = make_perturbation("resize", scale)[0](img, None)
        if jpeg_q < 100:
            img = make_perturbation("jpeg", int(jpeg_q))[0](img, None)
        scores, lines, heat_img = {}, [], None
        for name, d in dets.items():
            p = d.predict_pil(img)
            scores[f"{name}: AI-generated"] = p
            verdict = "AI-generated" if p >= d.threshold else "Real"
            lines.append(f"- **{name}**: P(AI-generated) = {p:.1%} → *{verdict}* (decision threshold {d.threshold:.2f})")
        if explain:
            d = dets.get("ResNet-18") or next(iter(dets.values()))
            x = d.preprocess(img).unsqueeze(0).to(d.device)
            rgb = denormalize(x[0], d.preprocess.mean, d.preprocess.std)
            if d.kind == "resnet":
                heat_img = overlay(rgb, gradcam_resnet(d.net, x), symmetric=False)
            else:
                heat_img = overlay(rgb, occlusion_map(d.prob_fn(), x, patch=56, stride=28, batch=16))
        agree = len({s >= 0.5 for s in scores.values()}) == 1
        lines.append("\nModels agree." if agree else "\n⚠️ The models disagree. Treat this result as uncertain.")
        return scores, heat_img, "\n".join(lines)

    with gr.Blocks(title="FakeLens") as demo:
        gr.Markdown("# FakeLens: is this image AI-generated?")
        gr.Markdown(DISCLAIMER)
        with gr.Row():
            with gr.Column():
                inp = gr.Image(type="pil", label="Upload an image")
                jpeg = gr.Slider(10, 100, value=100, step=1, label="Simulate JPEG quality (100 = off)")
                scale = gr.Slider(0.25, 2.0, value=1.0, step=0.05, label="Simulate resize (1.0 = off)")
                explain = gr.Checkbox(value=False, label="Show explanation heatmap (slower on CPU)")
                btn = gr.Button("Analyse", variant="primary")
            with gr.Column():
                out_lbl = gr.Label(num_top_classes=4, label="P(AI-generated) per model")
                out_txt = gr.Markdown()
                out_heat = gr.Image(label="Where the model looks (ResNet: Grad-CAM; CLIP: occlusion, red = supports 'fake')")
        btn.click(run, [inp, jpeg, scale, explain], [out_lbl, out_heat, out_txt])
    return demo


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--clip", default=os.environ.get("FAKELENS_CLIP_CKPT", "checkpoints/A2_head.pt"))
    ap.add_argument("--resnet", default=os.environ.get("FAKELENS_RESNET_CKPT", "checkpoints/B_model.pt"))
    ap.add_argument("--share", action="store_true")
    args = ap.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dets = {}
    if Path(args.clip).exists():
        dets["CLIP multi-layer (A2)"] = Detector(args.clip, device)
    if Path(args.resnet).exists():
        dets["ResNet-18"] = Detector(args.resnet, device)
    if not dets:
        raise SystemExit("No checkpoints found. Pass --clip/--resnet (copy head.pt / model.pt from runs/p1/...).")
    build_app(dets).launch(share=args.share)


if __name__ == "__main__":
    main()
