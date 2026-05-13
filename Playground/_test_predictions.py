import sys, numpy as np, cv2
sys.path.insert(0, '.')
import numpy_jepa_runtime as rt

model = rt.load_model()
print("Classes:", list(model.text_mapping.values()))
print()

def make_walking_video():
    frames = []
    base = np.ones((256, 256, 3), dtype=np.uint8) * 128
    for i in range(8):
        f = base.copy()
        y_offset = 50 + i * 12
        f[y_offset:y_offset+60, 100:150] = [200, 180, 160]
        frames.append(f)
    return np.array(frames)

def make_boxing_video():
    frames = []
    base = np.ones((256, 256, 3), dtype=np.uint8) * 128
    for i in range(8):
        f = base.copy()
        x = 100 if i % 2 == 0 else 160
        f[120:160, x:x+40] = [255, 200, 200]
        frames.append(f)
    return np.array(frames)

def make_clapping_video():
    frames = []
    base = np.ones((256, 256, 3), dtype=np.uint8) * 128
    for i in range(8):
        f = base.copy()
        offset = 5 if i % 2 == 0 else 0
        f[130:160, 90+offset:110+offset] = [220, 180, 180]
        f[130:160, 150-offset:170-offset] = [220, 180, 180]
        frames.append(f)
    return np.array(frames)

for name, video_fn in [("WALKING", make_walking_video), ("BOXING", make_boxing_video), ("CLAPPING", make_clapping_video)]:
    frames = video_fn()
    results = rt.predict_frames(model, frames, "describe action")
    print(name + ":")
    for r in results[:3]:
        label = r["label"]
        score = r["score"]
        print(f"  {label:<30s}  {score:.3f}")
    print()
