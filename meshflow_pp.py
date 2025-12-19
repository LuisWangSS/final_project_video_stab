import cv2
import math
import numpy as np
from collections import deque
from dataclasses import dataclass
import tqdm


@dataclass
class MeshFlowPPConfig:
    """
    配置参数，涵盖四个改进模块。
    """

    # 基础网格
    base_rows: int = 12
    base_cols: int = 12
    max_refine_depth: int = 1
    grad_trigger: float = 18.0
    residual_trigger: float = 1.2
    inlier_trigger: float = 0.35

    # 特征追踪与 FB 门控
    max_features: int = 1200
    feature_quality: float = 0.01
    min_distance: int = 5
    fb_cycle_thresh: float = 1.2

    # 在线 L1 平滑 (IRLS 近似)
    window: int = 32
    lambda_data: float = 1.5
    lambda_smooth: float = 3.0
    irls_iters: int = 6

    # 裁剪/缩放控制
    crop_margin: float = 0.92  # 可视区域保留比例
    crop_budget: float = 1.20  # 最大可用缩放倍数
    zoom_rate: float = 0.03    # 速率限制
    zoom_lpf: float = 0.4      # 一阶低通系数

    visualize: bool = False


class MeshFlowPP:
    """
    MeshFlow++：基于 MeshFlow 的在线改进版本，含 FB 门控、自适应网格、
    在线 L1 平滑与裁剪预算控制。
    """

    def __init__(self, config: MeshFlowPPConfig | None = None):
        self.cfg = config or MeshFlowPPConfig()
        self.base_mesh = None
        self.prev_frame = None
        self.prev_gray = None
        self.prev_disp = None
        self.obs_window = deque()  # 观测位移滑动窗口
        self.smoothed_history = []  # 保存平滑后的位移轨迹
        self.zoom = 1.0
        self.zoom_history = []
        self._D_cache = {}

    # ========= 高层接口 =========
    def stabilize(self, input_path: str, output_path: str):
        cap = cv2.VideoCapture(input_path)
        if not cap.isOpened():
            raise IOError(f"无法打开视频: {input_path}")

        num_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        codec = int(cap.get(cv2.CAP_PROP_FOURCC))

        ret, frame0 = cap.read()
        if not ret:
            raise IOError("视频为空或无法读取首帧")

        frame_h, frame_w = frame0.shape[:2]
        self.base_mesh = self._build_base_mesh(frame_w, frame_h)
        self.prev_frame = frame0
        self.prev_gray = cv2.cvtColor(frame0, cv2.COLOR_BGR2GRAY)
        self.prev_disp = np.zeros_like(self.base_mesh)
        self.obs_window.clear()
        self.obs_window.append(self.prev_disp.copy())
        self.smoothed_history.append(self.prev_disp.copy())
        self.zoom_history = [1.0]

        stabilized_frames = [frame0.copy()]

        pbar = tqdm.trange(num_frames - 1, desc="MeshFlow++")
        for _ in pbar:
            ret, frame = cap.read()
            if not ret:
                break
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            prev_pts, curr_pts = self._detect_and_track(self.prev_gray, gray)
            if prev_pts is None or curr_pts is None or len(prev_pts) < 6:
                # 特征不足，回退为零位移
                velocity = np.zeros_like(self.base_mesh)
            else:
                homography = self._estimate_homography(prev_pts, curr_pts)
                velocity = self._estimate_mesh_velocity(
                    frame, self.prev_frame, prev_pts, curr_pts, homography
                )

            current_disp = self.prev_disp + velocity

            # 在线 L1 平滑（IRLS 近似）
            smoothed_disp = self._smooth_window(current_disp, frame.shape[:2])
            self.smoothed_history.append(smoothed_disp)

            # 裁剪预算控制
            self.zoom = self._update_zoom(smoothed_disp, frame.shape[:2])
            self.zoom_history.append(self.zoom)

            stabilized_frame = self._warp_with_mesh(
                frame, self.base_mesh, smoothed_disp, self.zoom
            )
            stabilized_frames.append(stabilized_frame)

            if self.cfg.visualize:
                vis = np.hstack((self.prev_frame, stabilized_frame))
                cv2.imshow("MeshFlow++ (prev | stabilized)", vis)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

            # 滑动窗口维护
            self.obs_window.append(current_disp)
            if len(self.obs_window) > self.cfg.window:
                self.obs_window.popleft()

            self.prev_frame = frame
            self.prev_gray = gray
            self.prev_disp = current_disp

        cap.release()
        cv2.destroyAllWindows()

        self._write_video(output_path, stabilized_frames, fps, codec)
        metrics = self._summarize_metrics(frame_w, frame_h)
        return metrics

    # ========= 核心子模块 =========
    def _build_base_mesh(self, frame_w: int, frame_h: int):
        cols = self.cfg.base_cols
        rows = self.cfg.base_rows
        xs = np.linspace(0, frame_w - 1, cols + 1)
        ys = np.linspace(0, frame_h - 1, rows + 1)
        grid_x, grid_y = np.meshgrid(xs, ys)
        return np.stack([grid_x, grid_y], axis=-1).astype(np.float32)

    def _detect_and_track(self, prev_gray, curr_gray):
        pts0 = cv2.goodFeaturesToTrack(
            prev_gray,
            maxCorners=self.cfg.max_features,
            qualityLevel=self.cfg.feature_quality,
            minDistance=self.cfg.min_distance,
            blockSize=5,
        )
        if pts0 is None:
            return (None, None)

        pts1, st, _ = cv2.calcOpticalFlowPyrLK(prev_gray, curr_gray, pts0, None, maxLevel=3)
        if pts1 is None:
            return (None, None)

        st = st.reshape(-1).astype(bool)
        pts0_f = pts0.reshape(-1, 2)
        pts1_f = pts1.reshape(-1, 2)
        pts0_f = pts0_f[st]
        pts1_f = pts1_f[st]
        if len(pts0_f) == 0:
            return (None, None)

        # 前后向一致性门控
        pts0_back, st_back, _ = cv2.calcOpticalFlowPyrLK(curr_gray, prev_gray, pts1_f.reshape(-1, 1, 2), None, maxLevel=3)
        st_back = st_back.reshape(-1).astype(bool)
        valid = st_back
        pts0_back = pts0_back.reshape(-1, 2)
        cycle_err = np.linalg.norm(pts0_back - pts0_f, axis=1)
        valid = valid & (cycle_err < self.cfg.fb_cycle_thresh)

        if not np.any(valid):
            return (None, None)

        return (pts0_f[valid], pts1_f[valid])

    def _estimate_homography(self, prev_pts, curr_pts):
        H, _ = cv2.findHomography(prev_pts, curr_pts, method=cv2.RANSAC, ransacReprojThreshold=3.0)
        if H is None:
            H = np.eye(3, dtype=np.float32)
        return H

    def _estimate_mesh_velocity(self, curr_frame, prev_frame, prev_pts, curr_pts, homography):
        h, w = curr_frame.shape[:2]
        rows = self.cfg.base_rows
        cols = self.cfg.base_cols

        base_vertices = self.base_mesh.reshape(-1, 1, 2)
        warped_vertices = cv2.perspectiveTransform(base_vertices, homography)
        global_vel = (warped_vertices - base_vertices).reshape(rows + 1, cols + 1, 2)

        # 计算特征残差
        predicted_curr = cv2.perspectiveTransform(prev_pts.reshape(-1, 1, 2), homography).reshape(-1, 2)
        residuals = curr_pts - predicted_curr

        # 梯度图用于触发细化
        grad_map = self._cell_gradient_map(prev_frame)

        # 统计每个网格的残差
        cell_offsets = np.zeros((rows, cols, 2), dtype=np.float32)
        cell_counts = np.zeros((rows, cols), dtype=np.float32)

        cell_size_x = w / cols
        cell_size_y = h / rows
        for p, res in zip(prev_pts, residuals):
            c = min(cols - 1, max(0, int(p[0] / cell_size_x)))
            r = min(rows - 1, max(0, int(p[1] / cell_size_y)))
            cell_offsets[r, c] += res
            cell_counts[r, c] += 1

        # 自适应细化：若梯度或残差大，则 2x2 子网格计算局部残差
        vertex_offsets = np.zeros_like(global_vel)
        vertex_counts = np.zeros(global_vel.shape[:2], dtype=np.float32)

        for r in range(rows):
            for c in range(cols):
                if cell_counts[r, c] > 0:
                    cell_residual = cell_offsets[r, c] / cell_counts[r, c]
                else:
                    cell_residual = np.zeros(2, dtype=np.float32)

                trigger_grad = grad_map[r, c] > self.cfg.grad_trigger
                trigger_res = np.linalg.norm(cell_residual) > self.cfg.residual_trigger
                trigger_inlier = (cell_counts[r, c] / max(1.0, len(prev_pts))) < self.cfg.inlier_trigger
                refine = (self.cfg.max_refine_depth > 0) and (trigger_grad or trigger_res or trigger_inlier)

                if not refine:
                    self._accumulate_to_cell_vertices(vertex_offsets, vertex_counts, r, c, cell_residual)
                else:
                    # 2x2 子单元局部残差
                    sub_offsets = np.zeros((2, 2, 2), dtype=np.float32)
                    sub_counts = np.zeros((2, 2), dtype=np.float32)
                    sub_w = cell_size_x / 2
                    sub_h = cell_size_y / 2
                    for p, res in zip(prev_pts, residuals):
                        if not (c * cell_size_x <= p[0] < (c + 1) * cell_size_x and r * cell_size_y <= p[1] < (r + 1) * cell_size_y):
                            continue
                        sub_c = 0 if (p[0] - c * cell_size_x) < sub_w else 1
                        sub_r = 0 if (p[1] - r * cell_size_y) < sub_h else 1
                        sub_offsets[sub_r, sub_c] += res
                        sub_counts[sub_r, sub_c] += 1
                    for sr in range(2):
                        for sc in range(2):
                            if sub_counts[sr, sc] > 0:
                                sub_res = sub_offsets[sr, sc] / sub_counts[sr, sc]
                            else:
                                sub_res = cell_residual
                            self._accumulate_to_cell_vertices(vertex_offsets, vertex_counts, r, c, sub_res, sr, sc)

        vertex_offsets = np.divide(
            vertex_offsets,
            vertex_counts[..., None] + 1e-6,
        )
        return global_vel + vertex_offsets

    def _accumulate_to_cell_vertices(self, vertex_offsets, vertex_counts, r, c, residual, sub_r=0, sub_c=0):
        # 依据子单元位置稍微不同地分配权重，提升局部自由度
        weights = np.array([[0.55, 0.45], [0.45, 0.55]], dtype=np.float32)
        w_tl = weights[sub_r, sub_c]
        w_tr = 1 - w_tl
        w_bl = 1 - w_tl
        w_br = w_tl

        vertex_offsets[r, c] += w_tl * residual
        vertex_offsets[r, c + 1] += w_tr * residual
        vertex_offsets[r + 1, c] += w_bl * residual
        vertex_offsets[r + 1, c + 1] += w_br * residual

        vertex_counts[r, c] += w_tl
        vertex_counts[r, c + 1] += w_tr
        vertex_counts[r + 1, c] += w_bl
        vertex_counts[r + 1, c + 1] += w_br

    def _cell_gradient_map(self, frame_bgr):
        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
        gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
        mag = cv2.magnitude(gx, gy)
        rows = self.cfg.base_rows
        cols = self.cfg.base_cols
        cell_mag = np.zeros((rows, cols), dtype=np.float32)
        h, w = gray.shape
        cell_h = h // rows
        cell_w = w // cols
        for r in range(rows):
            for c in range(cols):
                block = mag[r * cell_h:(r + 1) * cell_h, c * cell_w:(c + 1) * cell_w]
                cell_mag[r, c] = float(np.mean(block))
        return cell_mag

    # ========= 在线 L1 平滑 =========
    def _smooth_window(self, latest_disp, frame_shape):
        self.obs_window[-1] = latest_disp
        obs = list(self.obs_window)
        T = len(obs)
        rows = self.cfg.base_rows + 1
        cols = self.cfg.base_cols + 1
        disp_stack = np.stack(obs, axis=0)  # [T, rows, cols, 2]

        # 对每个顶点的 x/y 分量分别执行 IRLS
        smoothed = np.zeros_like(disp_stack)
        for r in range(rows):
            for c in range(cols):
                for axis in range(2):
                    series = disp_stack[:, r, c, axis]
                    smoothed[:, r, c, axis] = self._irls_trend(series)

        # 仅返回窗口末尾的平滑结果
        last = smoothed[-1]

        # 约束：保持顶点位移在裁剪裕度内（粗略投影）
        h, w = frame_shape
        max_dx = (1 - self.cfg.crop_margin) * w * 0.5
        max_dy = (1 - self.cfg.crop_margin) * h * 0.5
        last[..., 0] = np.clip(last[..., 0], -max_dx, max_dx)
        last[..., 1] = np.clip(last[..., 1], -max_dy, max_dy)

        # 把平滑后的全窗口替换掉队列，保持历史一致
        new_obs = deque(maxlen=self.cfg.window)
        for k in range(T):
            new_obs.append(smoothed[k])
        self.obs_window = new_obs
        return last

    def _get_D(self, T: int):
        if T in self._D_cache:
            return self._D_cache[T]
        D = np.zeros((T - 2, T), dtype=np.float32)
        for i in range(T - 2):
            D[i, i] = 1.0
            D[i, i + 1] = -2.0
            D[i, i + 2] = 1.0
        self._D_cache[T] = D
        return D

    def _irls_trend(self, y: np.ndarray):
        y = y.astype(np.float32)
        T = len(y)
        if T <= 2:
            return y
        D = self._get_D(T)
        x = y.copy()
        lam_d = self.cfg.lambda_data
        lam_s = self.cfg.lambda_smooth
        eps = 1e-3
        for _ in range(self.cfg.irls_iters):
            w_data = lam_d / (np.abs(x - y) + eps)
            Dx = D @ x
            w_smooth = lam_s / (np.abs(Dx) + eps)
            A = np.diag(w_data) + D.T @ (np.diag(w_smooth) @ D)
            b = w_data * y
            x = np.linalg.solve(A, b)
        return x

    # ========= 裁剪预算控制 =========
    def _update_zoom(self, disp_last, frame_shape):
        h, w = frame_shape
        rows = self.cfg.base_rows
        cols = self.cfg.base_cols
        corners_idx = [(0, 0), (0, cols), (rows, 0), (rows, cols)]
        corners = np.array([self.base_mesh[r, c] + disp_last[r, c] for r, c in corners_idx], dtype=np.float32)

        min_xy = corners.min(axis=0)
        max_xy = corners.max(axis=0)

        span_x = max_xy[0] - min_xy[0]
        span_y = max_xy[1] - min_xy[1]
        s_req = max(span_x / (self.cfg.crop_margin * w), span_y / (self.cfg.crop_margin * h), 1.0)

        # 速率限制 + 低通
        delta = np.clip(s_req - self.zoom, -self.cfg.zoom_rate, self.cfg.zoom_rate)
        target = np.clip(self.zoom + delta, 1.0, self.cfg.crop_budget)
        smoothed = (1 - self.cfg.zoom_lpf) * self.zoom + self.cfg.zoom_lpf * target
        return smoothed

    # ========= 网格变形渲染 =========
    def _warp_with_mesh(self, frame, base_mesh, disp, zoom):
        h, w = frame.shape[:2]
        rows = self.cfg.base_rows
        cols = self.cfg.base_cols

        # 缩放围绕图像中心
        center = np.array([w / 2.0, h / 2.0], dtype=np.float32)
        stabilized_vertices = base_mesh + disp
        stabilized_vertices = center + (stabilized_vertices - center) / zoom

        # 构建稠密映射
        map_x = np.full((h, w), w + 1, dtype=np.float32)
        map_y = np.full((h, w), h + 1, dtype=np.float32)

        for r in range(rows):
            for c in range(cols):
                src = np.array([
                    base_mesh[r, c],
                    base_mesh[r, c + 1],
                    base_mesh[r + 1, c],
                    base_mesh[r + 1, c + 1],
                ], dtype=np.float32)
                dst = np.array([
                    stabilized_vertices[r, c],
                    stabilized_vertices[r, c + 1],
                    stabilized_vertices[r + 1, c],
                    stabilized_vertices[r + 1, c + 1],
                ], dtype=np.float32)
                H, _ = cv2.findHomography(dst, src)
                if H is None:
                    continue
                x_min = int(min(dst[:, 0].min(), src[:, 0].min()))
                x_max = int(max(dst[:, 0].max(), src[:, 0].max()))
                y_min = int(min(dst[:, 1].min(), src[:, 1].min()))
                y_max = int(max(dst[:, 1].max(), src[:, 1].max()))
                x_min = max(0, x_min - 2)
                y_min = max(0, y_min - 2)
                x_max = min(w - 1, x_max + 2)
                y_max = min(h - 1, y_max + 2)

                grid_x, grid_y = np.meshgrid(
                    np.arange(x_min, x_max + 1, dtype=np.float32),
                    np.arange(y_min, y_max + 1, dtype=np.float32),
                )
                pts = np.stack([grid_x, grid_y], axis=-1).reshape(-1, 1, 2)
                mapped = cv2.perspectiveTransform(pts, H).reshape(grid_y.shape + (2,))
                mask = np.ones_like(grid_x, dtype=bool)
                map_x[y_min:y_max + 1, x_min:x_max + 1] = np.where(mask, mapped[..., 0], map_x[y_min:y_max + 1, x_min:x_max + 1])
                map_y[y_min:y_max + 1, x_min:x_max + 1] = np.where(mask, mapped[..., 1], map_y[y_min:y_max + 1, x_min:x_max + 1])

        warped = cv2.remap(frame, map_x, map_y, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
        return warped

    # ========= 输出与指标 =========
    def _write_video(self, output_path, frames, fps, codec):
        h, w = frames[0].shape[:2]
        writer = cv2.VideoWriter(output_path, codec, fps, (w, h))
        for f in tqdm.tqdm(frames, desc="Writing"):
            writer.write(f)
        writer.release()

    def _summarize_metrics(self, frame_w, frame_h):
        zoom_arr = np.array(self.zoom_history)
        avg_crop_ratio = float(np.mean(1.0 / zoom_arr))
        # 简单稳定度指标：低频能量占比
        disp_stack = np.stack(self.smoothed_history, axis=0)  # [T, rows+1, cols+1, 2]
        vx = np.diff(disp_stack[..., 0], axis=0)
        vy = np.diff(disp_stack[..., 1], axis=0)
        fx = np.fft.fft(vx, axis=0)
        fy = np.fft.fft(vy, axis=0)
        fx_energy = np.square(np.abs(fx))
        fy_energy = np.square(np.abs(fy))
        total = fx_energy.sum(axis=0) + fy_energy.sum(axis=0) + 1e-6
        low = fx_energy[1:6].sum(axis=0) + fy_energy[1:6].sum(axis=0)
        stability = float(np.mean(low / total))
        return {
            "avg_crop_ratio": avg_crop_ratio,
            "stability_score": stability,
            "frames": len(self.smoothed_history),
        }


def demo():
    import argparse

    parser = argparse.ArgumentParser(description="MeshFlow++ 视频稳定")
    parser.add_argument("input", type=str, help="输入视频路径")
    parser.add_argument("output", type=str, nargs="?", help="输出视频路径（可选）")
    parser.add_argument("--visualize", action="store_true", help="实时预览")
    parser.add_argument("--base-rows", type=int, default=12)
    parser.add_argument("--base-cols", type=int, default=12)
    parser.add_argument("--window", type=int, default=32)
    parser.add_argument("--lambda-data", type=float, default=1.5)
    parser.add_argument("--lambda-smooth", type=float, default=3.0)
    parser.add_argument("--crop-margin", type=float, default=0.92)
    parser.add_argument("--crop-budget", type=float, default=1.2)

    args = parser.parse_args()

    output = args.output
    if output is None:
        import os
        base, ext = os.path.splitext(args.input)
        output = f"{base}-meshflowpp{ext}"

    cfg = MeshFlowPPConfig(
        base_rows=args.base_rows,
        base_cols=args.base_cols,
        window=args.window,
        lambda_data=args.lambda_data,
        lambda_smooth=args.lambda_smooth,
        crop_margin=args.crop_margin,
        crop_budget=args.crop_budget,
        visualize=args.visualize,
    )

    stabilizer = MeshFlowPP(cfg)
    metrics = stabilizer.stabilize(args.input, output)
    print("MeshFlow++ 完成")
    print(f"输出: {output}")
    print(f"平均裁剪比例: {metrics['avg_crop_ratio']:.4f}")
    print(f"稳定性分数: {metrics['stability_score']:.4f}")
    print(f"处理帧数: {metrics['frames']}")


if __name__ == "__main__":
    demo()

