import sys, numpy as np, cv2
sys.path.insert(0, '.')
import numpy_jepa_runtime as rt

model = rt.load_model()

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

# Analyze the optical flow features for each synthetic video
for name, video_fn in [("WALKING", make_walking_video), ("BOXING", make_boxing_video), ("CLAPPING", make_clapping_video)]:
    frames = video_fn()
    
    # Compute flow
    flow_vecs = []
    prev_gray = None
    for frame in frames:
        gray = cv2.cvtColor(frame.astype(np.uint8), cv2.COLOR_RGB2GRAY)
        if prev_gray is not None:
            flow = cv2.calcOpticalFlowFarneback(
                prev_gray, gray, None,
                pyr_scale=0.5, levels=3, winsize=15,
                iterations=3, poly_n=5, poly_sigma=1.2, flags=0,
            )
            flow_vecs.append(flow)
        prev_gray = gray

    flow_stack = np.stack(flow_vecs, axis=0)
    fx = flow_stack[..., 0]
    fy = flow_stack[..., 1]
    mag = np.sqrt(fx**2 + fy**2)

    mean_mag    = float(np.mean(mag))
    mean_abs_fx = float(np.mean(np.abs(fx)))
    mean_abs_fy = float(np.mean(np.abs(fy)))
    frame_mags  = mag.mean(axis=(1, 2))
    temp_var    = float(np.var(frame_mags))

    W = mag.shape[2]
    mid   = W // 2
    left  = mag[:, :, :mid]
    right = mag[:, :, W - mid:][:, :, ::-1]
    symmetry = 1.0 - float(
        np.mean(np.abs(left - right)) / (np.mean(left + right) + 1e-6)
    )
    symmetry = float(np.clip(symmetry, 0.0, 1.0))

    print(f"{name}:")
    print(f"  mean_mag={mean_mag:.4f}  mean_fx={mean_abs_fx:.4f}  mean_fy={mean_abs_fy:.4f}")
    print(f"  temp_var={temp_var:.4f}  symmetry={symmetry:.4f}")
    
    motion_scores = rt._motion_heuristic_scores(frames, model)
    print(f"  motion_scores: boxing={motion_scores[0]:.3f}  walking={motion_scores[1]:.3f}  clapping={motion_scores[2]:.3f}")
    print()
