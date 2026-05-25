import numpy as np

def fix_quaternion_signs(quaternions_xyzw: np.ndarray) -> np.ndarray:
    q = quaternions_xyzw.copy()
    for i in range(1, len(q)):
        if np.dot(q[i], q[i-1]) < 0:
            q[i] = -q[i]
    return q