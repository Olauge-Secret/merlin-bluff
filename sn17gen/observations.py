"""The measured colour line and the content of the last refine: r22sc's text and views on X1hi's gap-free sheet. General, numbers only, no object names."""
import io
import numpy as np
from PIL import Image
from . import measure_own as M

REFINE_TEXT = "REFINEMENT STEP {step} of {steps}: a chance to improve your module before it is judged. Below are the reference photo, renders of your current module and its code. The renders were made by the judge's own renderer from the four front angles the judge compares with the photo first, so every visible difference costs points.\n1. Compare the renders with the photo and list the concrete discrepancies, most damaging first: pose and orientation (which face points toward the front cameras; lying, standing, leaning or diagonal; tilt and turn), overall proportions (width : height : depth) and the sizes and positions of the parts, missing or extra parts (count every repeated part), colours and materials (hue, saturation, brightness, transparency, metal or gloss), and framing (the object alone fills the frame as in the photo: no floor, base, stand or backdrop). Ignore differences caused only by the plain white background and neutral light of the renders, and by the fit step that re-centres and rescales the object.\n2. Then write the COMPLETE corrected module that fixes these discrepancies, under the same module contract and rules. Keep every part that already matches the photo; do not simplify, drop correct detail or start over unless the object is fundamentally wrong. If nothing needs fixing, return the module unchanged.\n3. The corrected module must pass the same validator: only the allowed THREE members, THREE passed only in direct calls of named helpers (never to methods of helper objects), and the limits with a margin: added detail must keep the total under about 60,000 vertices, 120 draw calls and 4 MiB of texture data, because the renderer times out on heavy scenes (use 512x512 DataTextures: one 1024x1024 RGBA texture alone fills the 4 MiB).\nReply format for this step: first the discrepancy list (plain text, at most 12 short bullets, no code), then exactly one ```javascript fenced code block that contains the complete module."
REFERENCE_LABEL = 'Reference photo of the object:'
RENDERS_LABEL = "Renders of your current module by the judge's renderer (512 x 512 pixels, white background, neutral light), in this order: front-left 30 degrees, front-right 30 degrees, front from 15 degrees below, front from 30 degrees above:"
CODE_LABEL = 'Your current module:'


def _tiles(sheet: bytes) -> list[bytes]:
    im = Image.open(io.BytesIO(sheet)).convert("RGB")
    w, h = im.size
    out = []
    for y in (0, h // 2):
        for x in (0, w // 2):
            b = io.BytesIO()
            im.crop((x, y, x + w // 2, y + h // 2)).save(b, format="PNG", compress_level=6)
            out.append(b.getvalue())
    return out


def colour_note(photo: bytes, sheet: bytes) -> str:
    q = M.photo_mask(photo)
    if q is None:
        return ""
    pc = M.color(q[1], q[0])
    views = []
    for t in _tiles(sheet):
        rgb = np.asarray(Image.open(io.BytesIO(t)).resize((M.S, M.S), Image.BILINEAR)).astype(np.float32)
        m = M.render_mask(rgb)
        if m.sum() >= 30:
            views.append(M.color(rgb, m))
    if not views:
        return ""
    c, l = np.median([v["C90"] for v in views]), np.median([v["L"] for v in views])
    return (
        "Measured on the object's pixels, median of the four views (lightness 0-100; colourfulness of the strongest colours, higher is more vivid). Estimates: the photo's light differs from the judge's.\n"
        f"- colourfulness: photo {pc['C90']:.0f}, render {c:.0f}\n- lightness: photo {pc['L']:.0f}, render {l:.0f}\n"
        "Move material colours, textures, roughness and metalness toward the photo where the photo itself confirms these numbers.\n"
    )


def last_refine(photo: bytes, sheet: bytes, source: str, step: int, steps: int, data_url) -> list[dict]:
    def text(t):
        return {"type": "input_text", "text": t}

    def image(png):
        return {"type": "input_image", "image_url": data_url(png), "detail": "high"}
    note = colour_note(photo, sheet)
    return [
        *([text(note)] if note else []),
        text(REFINE_TEXT.format(step=step, steps=steps)),
        text(REFERENCE_LABEL),
        image(photo),
        text(RENDERS_LABEL),
        *(image(t) for t in _tiles(sheet)),
        text(f"{CODE_LABEL}\n```javascript\n{source.rstrip()}\n```"),
    ]
