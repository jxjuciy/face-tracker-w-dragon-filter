import os
import time
import math
import urllib.request

import cv2
import numpy as np
import mediapipe as mp
import moderngl
import trimesh
from mediapipe.tasks.python import BaseOptions
from mediapipe.tasks.python import vision
from PIL import Image

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
LANDMARKER_PATH = os.path.join(BASE_DIR, "face_landmarker.task")
LANDMARKER_URL = (
    "https://storage.googleapis.com/mediapipe-models/face_landmarker/"
    "face_landmarker/float16/1/face_landmarker.task"
)
MODEL_PATH = os.path.join(BASE_DIR, "model", "scene.gltf")
CAM_SOURCE = 0

# --- trigger (uses MediaPipe jawOpen score 0..1) ---
OPEN_SCORE = 0.35
CLOSE_SCORE = 0.18
FADE_SPEED = 0.2
SMOOTH = 0.5

# --- model fit (live keys can change these too) ---
# Rotate the GLTF bust into a head-on pose and anchor its face to the tracker.
MODEL_ROT_DEG = (0, 270, 0)
MODEL_WIDTH_CM = 36.0         # oversized bust covers the head and upper shoulders
MODEL_FACE_ANCHOR_FRACTION = 0.18  # face center above the bust's geometric center
MODEL_OFFSET_CM = [0.0, 1.0, 5.0]  # x right, y up, z toward camera
FLIP_UV = True                # set False if the texture looks scrambled

HIDE_FACE = True
FOV_Y = 63.0                  # MediaPipe's virtual camera

FACE_OVAL = [10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288, 397,
             365, 379, 378, 400, 377, 152, 148, 176, 149, 150, 136, 172, 58,
             132, 93, 234, 127, 162, 21, 54, 103, 67, 109]

VERT = """
#version 330
uniform mat4 mvp;
uniform mat4 model;
in vec3 in_pos; in vec3 in_norm; in vec2 in_uv;
out vec3 v_n; out vec3 v_p; out vec2 v_uv;
void main() {
    v_p = (model * vec4(in_pos, 1.0)).xyz;
    v_n = mat3(model) * in_norm;
    v_uv = in_uv;
    gl_Position = mvp * vec4(in_pos, 1.0);
}
"""

FRAG = """
#version 330
uniform sampler2D tex;
uniform bool use_tex;
uniform vec4 base_color;
uniform float gain;
in vec3 v_n; in vec3 v_p; in vec2 v_uv;
out vec4 f;
void main() {
    vec4 c = use_tex ? texture(tex, v_uv) * base_color : base_color;
    if (c.a < 0.1) discard;
    vec3 n = normalize(v_n);
    if (!gl_FrontFacing) n = -n;
    vec3 V = normalize(-v_p);
    vec3 L1 = normalize(vec3(0.4, 0.7, 0.6));   // key light
    vec3 L2 = normalize(vec3(-0.6, 0.1, 0.4));  // fill light
    float d = max(dot(n, L1), 0.0) * 0.9 + max(dot(n, L2), 0.0) * 0.35 + 0.18;
    float s = pow(max(dot(reflect(-L1, n), V), 0.0), 32.0) * 0.45;
    float rim = pow(1.0 - max(dot(n, V), 0.0), 3.0) * 0.4;
    vec3 col = c.rgb * d + s + rim * vec3(1.0, 0.55, 0.25);
    f = vec4(col * gain, 1.0);
}
"""


def ensure_model():
    if not os.path.exists(LANDMARKER_PATH):
        print("Downloading face model (one-time)...")
        urllib.request.urlretrieve(LANDMARKER_URL, LANDMARKER_PATH)
    if not os.path.exists(MODEL_PATH):
        raise RuntimeError(f"Dragon model not found at {MODEL_PATH}. Keep the model/ folder beside main.py.")


def euler(rx, ry, rz):
    rx, ry, rz = map(math.radians, (rx, ry, rz))
    X = np.array([[1, 0, 0], [0, math.cos(rx), -math.sin(rx)], [0, math.sin(rx), math.cos(rx)]])
    Y = np.array([[math.cos(ry), 0, math.sin(ry)], [0, 1, 0], [-math.sin(ry), 0, math.cos(ry)]])
    Z = np.array([[math.cos(rz), -math.sin(rz), 0], [math.sin(rz), math.cos(rz), 0], [0, 0, 1]])
    return Z @ Y @ X


# ---------- load + normalize the 3D model ----------
def load_parts():
    scene = trimesh.load(MODEL_PATH, force="scene")
    parts = []
    for node in scene.graph.nodes_geometry:
        T, gname = scene.graph[node]
        g = scene.geometry[gname]
        if not isinstance(g, trimesh.Trimesh) or len(g.faces) == 0:
            continue
        g = g.copy()
        g.apply_transform(T)
        f = g.faces
        part = {
            "pos": g.vertices[f].reshape(-1, 3).astype(np.float32),
            "nrm": g.vertex_normals[f].reshape(-1, 3).astype(np.float32),
            "uv": np.zeros((len(f) * 3, 2), np.float32),
            "img": None,
            "color": np.ones(4, np.float32),
        }
        vis = g.visual
        uv = getattr(vis, "uv", None)
        if uv is not None and len(uv) == len(g.vertices):
            part["uv"] = uv[f].reshape(-1, 2).astype(np.float32)
        mat = getattr(vis, "material", None)
        if mat is not None:
            img = getattr(mat, "baseColorTexture", None)
            if img is None:
                img = getattr(mat, "image", None)
            part["img"] = img
            bc = getattr(mat, "baseColorFactor", None)
            if bc is None:
                bc = getattr(mat, "diffuse", None)
            if bc is not None:
                bc = np.array(bc, np.float32)
                part["color"] = bc / 255.0 if bc.max() > 1.0 else bc
        elif getattr(vis, "kind", None) == "vertex":
            vc = vis.vertex_colors[f].reshape(-1, 4).mean(0) / 255.0
            part["color"] = vc.astype(np.float32)
        parts.append(part)
    if not parts:
        raise RuntimeError(f"No renderable meshes found in {MODEL_PATH}")

    # rotate, center, scale to a fixed width (in cm)
    R = euler(*MODEL_ROT_DEG).astype(np.float32)
    for p in parts:
        p["pos"] = p["pos"] @ R.T
        p["nrm"] = p["nrm"] @ R.T
    allp = np.vstack([p["pos"] for p in parts])
    lo, hi = allp.min(0), allp.max(0)
    center = (lo + hi) / 2
    s = MODEL_WIDTH_CM / max(hi[0] - lo[0], 1e-6)
    for p in parts:
        p["pos"] = (p["pos"] - center) * s
    lo, hi = (lo - center) * s, (hi - center) * s

    # The tracker transform is centered on the human face, not the whole bust.
    # Move the model origin to the dragon's face so its neck and shoulders extend
    # down over the user's upper body.
    face_anchor_y = MODEL_FACE_ANCHOR_FRACTION * (hi[1] - lo[1])
    for p in parts:
        p["pos"][:, 1] -= face_anchor_y

    return parts


# ---------- OpenGL renderer ----------
class Renderer:
    def __init__(self, parts, w, h):
        self.ctx = moderngl.create_standalone_context()
        self.prog = self.ctx.program(vertex_shader=VERT, fragment_shader=FRAG)
        self.w, self.h = w, h
        self.fbo = self.ctx.framebuffer(
            color_attachments=[self.ctx.texture((w, h), 4)],
            depth_attachment=self.ctx.depth_renderbuffer((w, h)))
        self.items = []
        for p in parts:
            data = np.hstack([p["pos"], p["nrm"], p["uv"]]).astype(np.float32)
            vbo = self.ctx.buffer(data.tobytes())
            vao = self.ctx.vertex_array(
                self.prog, [(vbo, "3f 3f 2f", "in_pos", "in_norm", "in_uv")])
            tex = None
            if p["img"] is not None:
                img = p["img"].convert("RGBA")
                if FLIP_UV:
                    img = img.transpose( Image.Transpose.FLIP_TOP_BOTTOM)  # OpenGL UV origin is bottom-left
                tex = self.ctx.texture(img.size, 4, img.tobytes())
                tex.build_mipmaps()
            self.items.append((p, vbo, vao, tex))
        f = 1 / math.tan(math.radians(FOV_Y) / 2)
        near, far = 1.0, 10000.0
        self.P = np.array([[f * h / w, 0, 0, 0], [0, f, 0, 0],
                           [0, 0, (far + near) / (near - far), 2 * far * near / (near - far)],
                           [0, 0, -1, 0]], np.float32)

    def render(self, model, gain):
        self.fbo.use()
        self.ctx.clear(0, 0, 0, 0)
        self.ctx.enable(moderngl.DEPTH_TEST)
        self.prog["mvp"].write((self.P @ model).T.astype(np.float32).tobytes())
        self.prog["model"].write(model.T.astype(np.float32).tobytes())
        self.prog["gain"].value = float(gain)
        for p, vbo, vao, tex in self.items:
            self.prog["base_color"].value = tuple(float(v) for v in p["color"][:4])
            self.prog["use_tex"].value = tex is not None
            if tex is not None:
                tex.use(0)
            vao.render()
        raw = self.fbo.read(components=4, alignment=1)
        img = np.frombuffer(raw, np.uint8).reshape(self.h, self.w, 4)[::-1]
        return cv2.cvtColor(img, cv2.COLOR_RGBA2BGRA)


# ---------- compositing ----------
def hide_face(frame, oval, opacity):
    h, w = frame.shape[:2]
    mask = np.zeros((h, w), np.uint8)
    cv2.fillPoly(mask, [np.int32(oval)], 255)
    face_width = max(float(np.ptp(oval[:, 0])), 1.0)
    expand = max(int(face_width * 0.5), 3) | 1
    mask = cv2.dilate(mask, np.ones((expand, expand), np.uint8))
    edge = max(int(face_width * 0.08), 3) | 1
    mask = cv2.GaussianBlur(mask, (edge, edge), 0).astype(np.float32) / 255

    blur_size = max(int(face_width * 0.9), 3) | 1
    max_kernel = min(h, w)
    if max_kernel % 2 == 0:
        max_kernel -= 1
    blur_size = min(blur_size, max_kernel)
    blurred = cv2.GaussianBlur(frame, (blur_size, blur_size), 0)
    m = (mask * opacity)[:, :, None]
    return (frame * (1 - m) + blurred * m).astype(np.uint8)


def composite(frame, layer, opacity):
    a = layer[:, :, 3:4].astype(np.float32) / 255 * opacity
    # soft contact shadow
    sh = cv2.GaussianBlur(layer[:, :, 3], (41, 41), 0).astype(np.float32) / 255
    sh = np.roll(sh, 12, axis=0)[:, :, None] * 0.4 * opacity
    out = frame.astype(np.float32) * (1 - sh)
    out = out * (1 - a) + layer[:, :, :3].astype(np.float32) * a
    return out.astype(np.uint8)


def translate(x, y, z):
    M = np.eye(4, dtype=np.float32)
    M[:3, 3] = (x, y, z)
    return M


def main():
    global MODEL_OFFSET_CM
    ensure_model()
    parts = load_parts()

    cap = cv2.VideoCapture(CAM_SOURCE, cv2.CAP_MSMF)
    if not cap.isOpened():
        cap = cv2.VideoCapture(CAM_SOURCE)
    if not cap.isOpened():
        raise RuntimeError("Could not open camera. Try CAM_SOURCE = 1.")
    ok, frame = cap.read()
    if not ok:
        raise RuntimeError("Camera gave no frame.")
    h, w = frame.shape[:2]
    renderer = Renderer(parts, w, h)

    landmarker = vision.FaceLandmarker.create_from_options(
        vision.FaceLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=LANDMARKER_PATH),
            running_mode=vision.RunningMode.VIDEO,
            num_faces=1,
            output_face_blendshapes=True,
            output_facial_transformation_matrixes=True,
        )
    )

    last_ts, is_open, opacity = 0, False, 0.0
    M_s, oval = None, None
    extra_scale, extra_yaw = 1.0, 0.0
    print("Open your mouth! Keys: W/S up/down, A/D forward/back, +/- size, "
          "R rotate 90, Q quit")

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            frame = cv2.flip(frame, 1)

            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB,
                                data=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            ts = max(int(time.time() * 1000), last_ts + 1)
            last_ts = ts
            res = landmarker.detect_for_video(mp_image, ts)

            if res.face_landmarks and res.facial_transformation_matrixes:
                M = np.array(res.facial_transformation_matrixes[0], np.float32)
                M_s = M if M_s is None else SMOOTH * M_s + (1 - SMOOTH) * M
                jaw = next((b.score for b in res.face_blendshapes[0]
                            if b.category_name == "jawOpen"), 0.0)
                if not is_open and jaw > OPEN_SCORE:
                    is_open = True
                elif is_open and jaw < CLOSE_SCORE:
                    is_open = False
                lm = res.face_landmarks[0]
                oval = np.array([[lm[i].x * w, lm[i].y * h] for i in FACE_OVAL], np.float32)
            else:
                is_open = False

            opacity = min(1.0, max(0.0, opacity + (FADE_SPEED if is_open else -FADE_SPEED)))

            if M_s is not None and opacity > 0:
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).mean()
                gain = float(np.clip(gray / 120, 0.6, 1.3))
                pop = 0.65 + 0.65 * opacity  # enlarge the bust as the mouth-triggered filter appears
                adj = translate(*MODEL_OFFSET_CM)
                R = np.eye(4, dtype=np.float32)
                R[:3, :3] = euler(0, extra_yaw, 0) * extra_scale * pop
                model = M_s @ adj @ R
                if HIDE_FACE and oval is not None:
                    frame = hide_face(frame, oval, opacity)
                layer = renderer.render(model, gain)
                frame = composite(frame, layer, opacity)

            cv2.imshow("Dragon Face 3D", frame)
            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                break
            elif key == ord("w"):
                MODEL_OFFSET_CM[1] += 0.5
            elif key == ord("s"):
                MODEL_OFFSET_CM[1] -= 0.5
            elif key == ord("a"):
                MODEL_OFFSET_CM[2] += 0.5
            elif key == ord("d"):
                MODEL_OFFSET_CM[2] -= 0.5
            elif key in (ord("+"), ord("=")):
                extra_scale *= 1.05
            elif key == ord("-"):
                extra_scale /= 1.05
            elif key == ord("r"):
                extra_yaw = (extra_yaw + 90) % 360
    finally:
        landmarker.close()
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
