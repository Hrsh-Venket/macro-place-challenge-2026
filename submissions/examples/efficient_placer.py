"""
EfficientPlace — Sequential macro placer with optional PPO training.

Implements the EfficientPlace method of Geng et al., ICML 2024
("Reinforcement Learning within Tree Search for Fast Macro Placement",
https://github.com/MIRALab-USTC/AI4EDA-EfficientPlace) including the
trained CNN actor-critic, not just the greedy inference policy.

Two phases:
  1. PPO **training** of a U-Net actor + critic on this benchmark, with
     reward = -delta_HPWL.  Each episode is a sequential placement of
     hard macros in a fixed ordering; the actor outputs a distribution
     over a discretized grid (action = grid cell index).
  2. Final **placement** combines:
       (a) trained-actor greedy rollouts (one per ordering),
       (b) baseline rank-based greedy rollouts (no policy),
       (c) the lowest-HPWL solution discovered during training.
     The lowest-HPWL valid placement wins.

Training time is controlled by `train_loops`, `episodes_per_loop`,
`update_epochs`, `batch_size`, and a hard `train_time_budget` wall-clock
cap (seconds).  Setting `train=False` skips training and reduces to the
greedy wire-mask inference policy alone.
"""

from __future__ import annotations

import math
import time
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical
from torch.utils.data.sampler import BatchSampler, SubsetRandomSampler

from macro_place.benchmark import Benchmark


# ──────────────────────────────────────────────────────────────────────────────
# Networks (ported from upstream src/agent.py — U-Net actor + 4-stage critic).
# Both expect grid divisible by 128 (upstream constraint).
# ──────────────────────────────────────────────────────────────────────────────


class _Actor(nn.Module):
    def __init__(self, num_steps: int, grid: int) -> None:
        super().__init__()
        self.grid = grid
        s = max(1, grid // 128)
        self.conv_1 = nn.Sequential(
            nn.AvgPool2d(s, s), nn.Conv2d(3, 8, 3, 1, 1), nn.ReLU()
        )
        self.pool_1 = nn.MaxPool2d(2, 2)
        self.conv_2 = nn.Sequential(nn.Conv2d(8, 16, 3, 1, 1), nn.ReLU())
        self.pool_2 = nn.MaxPool2d(4, 4)
        self.conv_3 = nn.Sequential(nn.Conv2d(16, 32, 3, 1, 1), nn.ReLU())
        self.pool_3 = nn.MaxPool2d(4, 4)
        self.conv_4 = nn.Sequential(nn.Conv2d(32, 32, 3, 1, 1), nn.ReLU())
        self.fc = nn.Sequential(
            nn.Linear(4 * 4 * 32 + 32, 512), nn.ReLU(),
            nn.Linear(512, 4 * 4 * 32), nn.ReLU(),
        )
        self.up_5 = nn.Upsample(scale_factor=4, mode="bilinear", align_corners=True)
        self.conv_5 = nn.Sequential(nn.Conv2d(64, 16, 3, 1, 1), nn.ReLU())
        self.up_6 = nn.Upsample(scale_factor=4, mode="bilinear", align_corners=True)
        self.conv_6 = nn.Sequential(nn.Conv2d(32, 8, 3, 1, 1), nn.ReLU())
        self.up_7 = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
        self.conv_7 = nn.Sequential(nn.Conv2d(16, 8, 3, 1, 1), nn.ReLU())
        self.conv_8 = nn.Sequential(
            nn.Upsample(scale_factor=s, mode="bilinear", align_corners=True),
            nn.Conv2d(8, 1, 1, 1),
        )
        self.time_embedding = nn.Embedding(num_steps, 32)

    def forward(self, s: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        f1 = self.conv_1(s)
        f1p = self.pool_1(f1)
        f2 = self.conv_2(f1p)
        f2p = self.pool_2(f2)
        f3 = self.conv_3(f2p)
        f3p = self.pool_3(f3)
        f4 = self.conv_4(f3p)
        te = self.time_embedding(t)
        feat = torch.cat([f4.reshape(-1, 4 * 4 * 32), te], dim=-1)
        feat = self.fc(feat).reshape(-1, 32, 4, 4)
        f5 = self.conv_5(torch.cat((f3, self.up_5(feat)), dim=1))
        f6 = self.conv_6(torch.cat((f2, self.up_6(f5)), dim=1))
        f7 = self.conv_7(torch.cat((f1, self.up_7(f6)), dim=1))
        f8 = self.conv_8(f7)
        return f8.reshape(-1, self.grid * self.grid)

    def distr(
        self,
        logits: torch.Tensor,
        position_mask: torch.Tensor,
        wire_mask: torch.Tensor,
    ) -> Categorical:
        # Match upstream get_distr: only argmin-wire-mask cells get non-zero
        # probability — actor learns to pick among ties.
        flat_pos = position_mask.reshape(-1, self.grid * self.grid) >= 0.5
        masked_wire = torch.where(
            flat_pos, torch.full_like(wire_mask.reshape_as(flat_pos.float()), float("inf")),
            wire_mask.reshape(-1, self.grid * self.grid),
        )
        wire_min = masked_wire.min(dim=-1, keepdim=True)[0]
        greedy_mask = torch.where(masked_wire == wire_min, 0.0, -1e10)
        pi = torch.where(flat_pos, torch.full_like(logits, -1e10), logits)
        return Categorical(torch.softmax(pi + greedy_mask, dim=-1))


class _Critic(nn.Module):
    def __init__(self, num_steps: int, grid: int) -> None:
        super().__init__()
        s = max(1, grid // 128)
        self.conv_1 = nn.Sequential(
            nn.AvgPool2d(s, s), nn.Conv2d(3, 8, 3, 1, 1), nn.GELU()
        )
        self.pool_1 = nn.MaxPool2d(2, 2)
        self.conv_2 = nn.Sequential(nn.Conv2d(8, 16, 3, 1, 1), nn.GELU())
        self.pool_2 = nn.MaxPool2d(4, 4)
        self.conv_3 = nn.Sequential(nn.Conv2d(16, 32, 3, 1, 1), nn.GELU())
        self.pool_3 = nn.MaxPool2d(4, 4)
        self.conv_4 = nn.Sequential(nn.Conv2d(32, 32, 3, 1, 1), nn.GELU())
        self.fc1 = nn.Sequential(
            nn.Linear(4 * 4 * 32 + 32, 512), nn.Tanh(), nn.Linear(512, 512)
        )
        self.fc2 = nn.Sequential(nn.Linear(512, 512), nn.Tanh(), nn.Linear(512, 1))
        self.time_embedding = nn.Embedding(num_steps + 1, 32)
        self.time_value = nn.Embedding(num_steps + 1, 1)

    def forward(self, s: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        f1 = self.conv_1(s)
        f1p = self.pool_1(f1)
        f2 = self.conv_2(f1p)
        f2p = self.pool_2(f2)
        f3 = self.conv_3(f2p)
        f3p = self.pool_3(f3)
        f4 = self.conv_4(f3p)
        te = self.time_embedding(t)
        feat = torch.cat([f4.reshape(-1, 4 * 4 * 32), te], dim=-1)
        return self.fc2(self.fc1(feat)) + self.time_value(t)


# ──────────────────────────────────────────────────────────────────────────────
# Replay buffer (rollout storage for PPO).
# ──────────────────────────────────────────────────────────────────────────────


class _ReplayBuffer:
    def __init__(self, capacity: int, grid: int) -> None:
        self.s = torch.zeros([capacity, 3, grid, grid])
        self.t = torch.zeros([capacity], dtype=torch.long)
        self.a = torch.zeros([capacity, 1], dtype=torch.long)
        self.a_logp = torch.zeros([capacity, 1])
        self.r = torch.zeros([capacity, 1])
        self.done = torch.zeros([capacity, 1])
        self.capacity = capacity
        self.count = 0

    def store(self, s, t, a, a_logp, r, done):
        if self.count >= self.capacity:
            return
        self.s[self.count] = s.cpu()
        self.t[self.count] = t
        self.a[self.count] = a
        self.a_logp[self.count] = a_logp
        self.r[self.count] = r
        self.done[self.count] = done
        self.count += 1

    def get(self, device: torch.device):
        n = self.count
        out = (
            self.s[:n].to(device),
            self.t[:n].to(device),
            self.a[:n].to(device),
            self.a_logp[:n].to(device),
            self.r[:n].to(device),
            self.done[:n].to(device),
        )
        self.count = 0
        return out


# ──────────────────────────────────────────────────────────────────────────────
# Sequential placement environment (canvas / wire_mask / position_mask).
# Operates on a fixed grid of size G; actions are flat grid indices.
# All overlap and HPWL bookkeeping is done in continuous (micron) coordinates
# for accuracy, while the CNN observes a [3, G, G] tensor.
# ──────────────────────────────────────────────────────────────────────────────


class _PlaceEnv:
    def __init__(
        self,
        benchmark: Benchmark,
        ordering: List[int],
        net_meta: List[Tuple[np.ndarray, np.ndarray]],
        grid: int,
        gap: float,
        wire_mask_scale: float = 1e4,
        reward_scale: float = 1e3,
    ) -> None:
        self.benchmark = benchmark
        self.ordering = ordering
        self.net_meta = net_meta
        self.G = grid
        self.gap = gap
        self.wire_mask_scale = wire_mask_scale
        self.reward_scale = reward_scale

        n_hard = benchmark.num_hard_macros
        n_total = benchmark.num_macros
        num_ports = int(benchmark.port_positions.shape[0])

        self.cw = float(benchmark.canvas_width)
        self.ch = float(benchmark.canvas_height)
        self.gx_centers = (np.arange(grid) + 0.5) * (self.cw / grid)
        self.gy_centers = (np.arange(grid) + 0.5) * (self.ch / grid)
        self.cell_w = self.cw / grid
        self.cell_h = self.ch / grid

        self.sizes_np = benchmark.macro_sizes.numpy().astype(np.float64)
        self.fixed_np = benchmark.macro_fixed.numpy()
        self.init_pos_np = benchmark.macro_positions.numpy().astype(np.float64).copy()
        self.port_pos_np = (
            benchmark.port_positions.numpy().astype(np.float64)
            if num_ports > 0
            else np.zeros((0, 2), dtype=np.float64)
        )

        self.n_hard = n_hard
        self.n_total = n_total
        self.num_ports = num_ports
        self.n_steps = len(ordering)

        # owner→nets and (owner,net)→pin offsets
        self.owner_to_nets: List[List[int]] = [[] for _ in range(n_total + num_ports)]
        self.owner_pins_on_net: Dict[Tuple[int, int], List[np.ndarray]] = {}
        for net_id, (owners, offsets) in enumerate(net_meta):
            for e in range(owners.shape[0]):
                o = int(owners[e])
                if 0 <= o < n_total + num_ports:
                    self.owner_to_nets[o].append(net_id)
                    self.owner_pins_on_net.setdefault((o, net_id), []).append(offsets[e])

    # ----- helpers --------------------------------------------------------------

    def _owner_center(self, o: int) -> np.ndarray:
        if o < self.n_total:
            return self.positions_np[o]
        return self.port_pos_np[o - self.n_total]

    def _seed_net_bboxes(self) -> None:
        n_nets = self.benchmark.num_nets
        self.net_min = np.full((n_nets, 2), np.inf, dtype=np.float64)
        self.net_max = np.full((n_nets, 2), -np.inf, dtype=np.float64)
        self.net_count = np.zeros(n_nets, dtype=np.int64)
        for net_id, (owners, offsets) in enumerate(self.net_meta):
            for e in range(owners.shape[0]):
                o = int(owners[e])
                if not self.anchored[o]:
                    continue
                p = self._owner_center(o) + offsets[e]
                if self.net_count[net_id] == 0:
                    self.net_min[net_id] = p
                    self.net_max[net_id] = p
                else:
                    if p[0] < self.net_min[net_id, 0]: self.net_min[net_id, 0] = p[0]
                    if p[1] < self.net_min[net_id, 1]: self.net_min[net_id, 1] = p[1]
                    if p[0] > self.net_max[net_id, 0]: self.net_max[net_id, 0] = p[0]
                    if p[1] > self.net_max[net_id, 1]: self.net_max[net_id, 1] = p[1]
                self.net_count[net_id] += 1

    def _compute_pos_mask(self, macro_idx: int) -> Tuple[np.ndarray, np.ndarray, float, float]:
        mw = self.sizes_np[macro_idx, 0]
        mh = self.sizes_np[macro_idx, 1]
        half_w = mw / 2.0
        half_h = mh / 2.0
        valid_x = (self.gx_centers >= half_w - 1e-9) & (self.gx_centers <= self.cw - half_w + 1e-9)
        valid_y = (self.gy_centers >= half_h - 1e-9) & (self.gy_centers <= self.ch - half_h + 1e-9)
        canvas_ok = valid_x[:, None] & valid_y[None, :]
        if self.placed_hards:
            ph_pos = self.positions_np[self.placed_hards]
            ph_size = self.sizes_np[self.placed_hards]
            tx = (mw + ph_size[:, 0]) / 2.0 + self.gap
            ty = (mh + ph_size[:, 1]) / 2.0 + self.gap
            dx = np.abs(self.gx_centers[:, None] - ph_pos[None, :, 0])
            dy = np.abs(self.gy_centers[:, None] - ph_pos[None, :, 1])
            x_block = (dx < tx[None, :]).astype(np.int32)
            y_block = (dy < ty[None, :]).astype(np.int32)
            pos_mask = (x_block @ y_block.T) > 0
        else:
            pos_mask = np.zeros((self.G, self.G), dtype=bool)
        infeasible = pos_mask | (~canvas_ok)
        return infeasible, canvas_ok, mw, mh

    def _compute_wire_mask(self, macro_idx: int) -> np.ndarray:
        wire = np.zeros((self.G, self.G), dtype=np.float64)
        for net_id in self.owner_to_nets[macro_idx]:
            if self.net_count[net_id] == 0:
                continue
            bmin0, bmin1 = self.net_min[net_id, 0], self.net_min[net_id, 1]
            bmax0, bmax1 = self.net_max[net_id, 0], self.net_max[net_id, 1]
            for off in self.owner_pins_on_net[(macro_idx, net_id)]:
                px = self.gx_centers + off[0]
                py = self.gy_centers + off[1]
                dx = np.maximum(px - bmax0, 0.0) + np.maximum(bmin0 - px, 0.0)
                dy = np.maximum(py - bmax1, 0.0) + np.maximum(bmin1 - py, 0.0)
                wire += dx[:, None] + dy[None, :]
        return wire

    def _build_canvas(self) -> np.ndarray:
        # Indicator of placed-macro footprints on the grid (for the CNN to see).
        canvas = np.zeros((self.G, self.G), dtype=np.float32)
        for i in self.placed_hards:
            cx, cy = self.positions_np[i]
            half_w = self.sizes_np[i, 0] / 2.0
            half_h = self.sizes_np[i, 1] / 2.0
            x0 = max(0, int(np.floor((cx - half_w) / self.cell_w)))
            x1 = min(self.G, int(np.ceil((cx + half_w) / self.cell_w)))
            y0 = max(0, int(np.floor((cy - half_h) / self.cell_h)))
            y1 = min(self.G, int(np.ceil((cy + half_h) / self.cell_h)))
            canvas[x0:x1, y0:y1] = 1.0
        return canvas

    def _state_tensor(self) -> torch.Tensor:
        canvas = self._build_canvas()
        if self.t < self.n_steps:
            macro_idx = self.ordering[self.t]
            infeasible, _, _, _ = self._compute_pos_mask(macro_idx)
            wire = self._compute_wire_mask(macro_idx) / self.wire_mask_scale
        else:
            infeasible = np.ones((self.G, self.G), dtype=bool)
            wire = np.zeros((self.G, self.G), dtype=np.float64)
        return torch.from_numpy(
            np.stack([canvas, wire.astype(np.float32), infeasible.astype(np.float32)])
        )

    # ----- env API -------------------------------------------------------------

    def reset(self) -> torch.Tensor:
        self.t = 0
        self.positions_np = self.init_pos_np.copy()
        self.anchored = np.zeros(self.n_total + self.num_ports, dtype=bool)
        self.anchored[self.n_hard:self.n_total] = True
        self.anchored[self.n_total:] = True
        for i in range(self.n_hard):
            if self.fixed_np[i]:
                self.anchored[i] = True
        # Hard macros not in `ordering` are also implicitly anchored at init pos.
        ord_set = set(self.ordering)
        for i in range(self.n_hard):
            if i not in ord_set:
                self.anchored[i] = True
        self.placed_hards = [i for i in range(self.n_hard) if self.anchored[i]]
        self._seed_net_bboxes()
        self._cur_state = self._state_tensor()
        return self._cur_state

    def step(self, action: int) -> Tuple[torch.Tensor, float, bool, dict]:
        macro_idx = self.ordering[self.t]
        gx, gy = divmod(int(action), self.G)
        cx = float(np.clip(self.gx_centers[gx], self.sizes_np[macro_idx, 0] / 2.0,
                           self.cw - self.sizes_np[macro_idx, 0] / 2.0))
        cy = float(np.clip(self.gy_centers[gy], self.sizes_np[macro_idx, 1] / 2.0,
                           self.ch - self.sizes_np[macro_idx, 1] / 2.0))

        # delta_HPWL from current bboxes given pin placement at (cx, cy).
        delta = 0.0
        for net_id in self.owner_to_nets[macro_idx]:
            for off in self.owner_pins_on_net[(macro_idx, net_id)]:
                p0 = cx + off[0]; p1 = cy + off[1]
                if self.net_count[net_id] == 0:
                    self.net_min[net_id, 0] = p0; self.net_min[net_id, 1] = p1
                    self.net_max[net_id, 0] = p0; self.net_max[net_id, 1] = p1
                else:
                    if p0 < self.net_min[net_id, 0]:
                        delta += self.net_min[net_id, 0] - p0
                        self.net_min[net_id, 0] = p0
                    if p1 < self.net_min[net_id, 1]:
                        delta += self.net_min[net_id, 1] - p1
                        self.net_min[net_id, 1] = p1
                    if p0 > self.net_max[net_id, 0]:
                        delta += p0 - self.net_max[net_id, 0]
                        self.net_max[net_id, 0] = p0
                    if p1 > self.net_max[net_id, 1]:
                        delta += p1 - self.net_max[net_id, 1]
                        self.net_max[net_id, 1] = p1
                self.net_count[net_id] += 1

        self.positions_np[macro_idx, 0] = cx
        self.positions_np[macro_idx, 1] = cy
        self.anchored[macro_idx] = True
        self.placed_hards.append(macro_idx)

        reward = -delta / self.reward_scale
        self.t += 1
        done = self.t >= self.n_steps
        self._cur_state = self._state_tensor()
        return self._cur_state, reward, done, {"delta_hpwl": delta}


# ──────────────────────────────────────────────────────────────────────────────
# PPO agent (port of upstream Agent.update with a Categorical policy).
# ──────────────────────────────────────────────────────────────────────────────


class _Agent:
    def __init__(
        self,
        num_steps: int,
        grid: int,
        device: torch.device,
        lr_actor: float,
        lr_critic: float,
        actor_lr_anneal: float,
        critic_lr_anneal: float,
        clip_epsilon: float,
        entropy_coef: float,
        max_grad_norm: float,
        gamma: float,
    ) -> None:
        self.actor = _Actor(num_steps, grid).to(device)
        self.critic = _Critic(num_steps, grid).to(device)
        self.actor_opt = torch.optim.Adam(self.actor.parameters(), lr_actor)
        self.critic_opt = torch.optim.Adam(self.critic.parameters(), lr_critic)
        self.actor_sched = torch.optim.lr_scheduler.ExponentialLR(self.actor_opt, actor_lr_anneal)
        self.critic_sched = torch.optim.lr_scheduler.ExponentialLR(self.critic_opt, critic_lr_anneal)
        self.device = device
        self.grid = grid
        self.clip_epsilon = clip_epsilon
        self.entropy_coef = entropy_coef
        self.max_grad_norm = max_grad_norm
        self.gamma = gamma

    @torch.no_grad()
    def act(self, state: torch.Tensor, t: int, deterministic: bool = False) -> Tuple[int, float]:
        s = state.unsqueeze(0).to(self.device)
        tt = torch.tensor([t], dtype=torch.long, device=self.device)
        self.actor.eval()
        logits = self.actor(s, tt)
        distr = self.actor.distr(logits, position_mask=s[:, 2], wire_mask=s[:, 1])
        if deterministic:
            action = distr.probs.argmax(dim=-1)
        else:
            action = distr.sample()
        return int(action.item()), float(distr.log_prob(action).item())

    def update(
        self,
        buffer: _ReplayBuffer,
        update_epochs: int,
        batch_size: int,
    ) -> Dict[str, float]:
        s, t, a, a_logp, r, done = buffer.get(self.device)
        if s.shape[0] < 2:
            return {"actor_loss": 0.0, "critic_loss": 0.0, "entropy": 0.0}

        # Discounted returns over per-episode segments (done resets target to 0).
        with torch.no_grad():
            v_target = torch.zeros_like(r)
            target = 0.0
            for i in range(s.shape[0] - 1, -1, -1):
                if bool(done[i].item()):
                    target = 0.0
                target = r[i].item() + self.gamma * target
                v_target[i] = target
            v_pred = self.critic(s, t)
            adv = v_target - v_pred
            adv = (adv - adv.mean()) / (adv.std() + 1e-5)

        actor_loss_avg = critic_loss_avg = entropy_avg = 0.0
        n_updates = 0
        self.actor.train(); self.critic.train()
        for _ in range(update_epochs):
            for idx in BatchSampler(SubsetRandomSampler(range(s.shape[0])), batch_size, False):
                logits = self.actor(s[idx], t[idx])
                distr = self.actor.distr(logits, position_mask=s[idx][:, 2], wire_mask=s[idx][:, 1])
                new_logp = distr.log_prob(a[idx].squeeze(-1)).unsqueeze(-1)
                entropy = distr.entropy().mean()
                ratio = torch.exp(torch.clamp(new_logp - a_logp[idx], max=7.0))
                surr1 = ratio * adv[idx]
                surr2 = torch.clamp(ratio, 1 - self.clip_epsilon, 1 + self.clip_epsilon) * adv[idx]
                actor_loss = -torch.min(surr1, surr2).mean() - self.entropy_coef * entropy
                self.actor_opt.zero_grad()
                actor_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.actor.parameters(), self.max_grad_norm)
                self.actor_opt.step()

                v_s = self.critic(s[idx], t[idx])
                critic_loss = F.smooth_l1_loss(v_s, v_target[idx])
                self.critic_opt.zero_grad()
                critic_loss.backward()
                nn.utils.clip_grad_norm_(self.critic.parameters(), self.max_grad_norm)
                self.critic_opt.step()

                self.actor_sched.step(); self.critic_sched.step()
                actor_loss_avg += float(actor_loss.item())
                critic_loss_avg += float(critic_loss.item())
                entropy_avg += float(entropy.item())
                n_updates += 1
        n_updates = max(1, n_updates)
        return {
            "actor_loss": actor_loss_avg / n_updates,
            "critic_loss": critic_loss_avg / n_updates,
            "entropy": entropy_avg / n_updates,
        }


# ──────────────────────────────────────────────────────────────────────────────
# Top-level placer.
# ──────────────────────────────────────────────────────────────────────────────


class EfficientPlacer:
    """EfficientPlace: PPO-trained sequential macro placer."""

    def __init__(
        self,
        # ── Placement-time grid & legalization ──
        grid_size: int = 512,
        num_starts: int = 3,
        seed: int = 0,
        overlap_gap: float = 1e-3,
        radial_steps: int = 200,
        radial_angles: int = 16,
        # ── Training switches ──
        train: bool = True,
        train_grid_size: int = 128,        # CNN grid (must be 128, 256, 512, …)
        train_loops: int = 20,             # outer PPO loops
        episodes_per_loop: int = 4,
        update_epochs: int = 4,
        batch_size: int = 64,
        max_train_macros: int = 64,        # cap macros placed by the policy per episode
        train_time_budget: float = 600.0,  # hard wall-clock cap (seconds)
        # ── PPO hyperparameters ──
        lr_actor: float = 4e-4,
        lr_critic: float = 1e-3,
        actor_lr_anneal: float = 0.9999,
        critic_lr_anneal: float = 0.9999,
        clip_epsilon: float = 0.2,
        entropy_coef: float = 1e-3,
        max_grad_norm: float = 0.5,
        gamma: float = 1.0,
        # ── Compute ──
        device: str = "auto",
        verbose: bool = False,
    ) -> None:
        self.grid_size = grid_size
        self.num_starts = num_starts
        self.seed = seed
        self.overlap_gap = overlap_gap
        self.radial_steps = radial_steps
        self.radial_angles = radial_angles

        self.train = train
        self.train_grid_size = train_grid_size
        self.train_loops = train_loops
        self.episodes_per_loop = episodes_per_loop
        self.update_epochs = update_epochs
        self.batch_size = batch_size
        self.max_train_macros = max_train_macros
        self.train_time_budget = train_time_budget

        self.lr_actor = lr_actor
        self.lr_critic = lr_critic
        self.actor_lr_anneal = actor_lr_anneal
        self.critic_lr_anneal = critic_lr_anneal
        self.clip_epsilon = clip_epsilon
        self.entropy_coef = entropy_coef
        self.max_grad_norm = max_grad_norm
        self.gamma = gamma

        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)
        self.verbose = verbose

    # ──────────────────────────────────────────────────────────────────── place

    def place(self, benchmark: Benchmark) -> torch.Tensor:
        rng = np.random.default_rng(self.seed)
        torch.manual_seed(self.seed)

        n_hard = benchmark.num_hard_macros
        positions = benchmark.macro_positions.clone()
        if n_hard == 0:
            return positions

        orderings = self._build_orderings(benchmark, n_hard)
        net_meta = self._build_net_metadata(benchmark)

        candidates: List[Tuple[float, np.ndarray]] = []

        # ── Baseline: rank-based greedy on the placement grid ──
        for order in orderings[: self.num_starts]:
            cand_positions, cand_cost = self._run_greedy(
                benchmark, order, net_meta, rng, grid=self.grid_size
            )
            candidates.append((cand_cost, cand_positions))
            if self.verbose:
                print(f"  [greedy] HPWL={cand_cost:.2f}")

        # ── Optional PPO training ──
        if self.train and n_hard > 1:
            train_order = orderings[0][: min(self.max_train_macros, n_hard)]
            best_train_pos, best_train_hpwl = self._train(
                benchmark, train_order, net_meta
            )
            if best_train_pos is not None:
                # Trained-policy rollouts placed only `train_order` macros at
                # train_grid_size; remaining macros stay at init. Re-run greedy on the
                # full ordering at full grid_size starting from those positions —
                # i.e. fix the trained ones and let greedy fill the rest.
                refined_pos, refined_cost = self._refine_with_trained(
                    benchmark, orderings[0], net_meta, best_train_pos, train_order, rng
                )
                candidates.append((refined_cost, refined_pos))
                if self.verbose:
                    print(f"  [trained-refined] HPWL={refined_cost:.2f}")

        # ── Pick the best ──
        best_cost, best_positions = min(candidates, key=lambda x: x[0])
        positions[:n_hard] = torch.from_numpy(best_positions).float()
        return positions

    # ─────────────────────────────────────────────────────────────── orderings

    def _build_orderings(self, benchmark: Benchmark, n_hard: int) -> List[List[int]]:
        sizes = benchmark.macro_sizes[:n_hard].numpy().astype(np.float64)
        fixed = benchmark.macro_fixed[:n_hard].numpy()
        areas = sizes[:, 0] * sizes[:, 1]

        net_area_sum = np.zeros(n_hard, dtype=np.float64)
        degree = np.zeros(n_hard, dtype=np.int64)
        for net in benchmark.net_nodes:
            nodes = net.numpy()
            hards = nodes[nodes < n_hard]
            if hards.size <= 1:
                continue
            tot = float(areas[hards].sum())
            for nd in hards:
                net_area_sum[nd] += tot
                degree[nd] += 1

        movable = [i for i in range(n_hard) if not fixed[i]]
        order_area = sorted(movable, key=lambda i: (-areas[i], -net_area_sum[i]))
        order_areasum = sorted(movable, key=lambda i: -net_area_sum[i])
        order_degree = sorted(movable, key=lambda i: (-degree[i], -areas[i]))
        return [order_area, order_areasum, order_degree]

    # ─────────────────────────────────────────────────────────── net metadata

    def _build_net_metadata(self, benchmark: Benchmark) -> List[Tuple[np.ndarray, np.ndarray]]:
        nets: List[Tuple[np.ndarray, np.ndarray]] = []
        if len(benchmark.net_pin_nodes) > 0:
            n_hard = benchmark.num_hard_macros
            pin_offsets_np = [
                po.numpy().astype(np.float64) if po.numel() > 0 else np.zeros((0, 2))
                for po in benchmark.macro_pin_offsets
            ]
            for net_pn in benchmark.net_pin_nodes:
                if net_pn.numel() == 0:
                    nets.append((np.zeros(0, dtype=np.int64), np.zeros((0, 2))))
                    continue
                owners = net_pn[:, 0].numpy().astype(np.int64)
                pin_idx = net_pn[:, 1].numpy().astype(np.int64)
                offsets = np.zeros((owners.shape[0], 2), dtype=np.float64)
                for e in range(owners.shape[0]):
                    o = int(owners[e]); p = int(pin_idx[e])
                    if o < n_hard and p < pin_offsets_np[o].shape[0]:
                        offsets[e] = pin_offsets_np[o][p]
                nets.append((owners, offsets))
        else:
            for net in benchmark.net_nodes:
                owners = net.numpy().astype(np.int64)
                offsets = np.zeros((owners.shape[0], 2), dtype=np.float64)
                nets.append((owners, offsets))
        return nets

    # ───────────────────────────────────────────────────────── radial fallback

    def _radial_search(
        self,
        target_x: float, target_y: float,
        mw: float, mh: float, half_w: float, half_h: float,
        cw: float, ch: float,
        placed_hards: List[int],
        positions_np: np.ndarray, sizes_np: np.ndarray, gap: float,
    ) -> Tuple[float, float]:
        if not placed_hards:
            return target_x, target_y
        ph_pos = positions_np[placed_hards]
        ph_size = sizes_np[placed_hards]
        tx = (mw + ph_size[:, 0]) / 2.0 + gap
        ty = (mh + ph_size[:, 1]) / 2.0 + gap

        def overlap_area(cx: float, cy: float) -> float:
            ox = np.maximum(tx - np.abs(cx - ph_pos[:, 0]), 0.0)
            oy = np.maximum(ty - np.abs(cy - ph_pos[:, 1]), 0.0)
            return float((ox * oy).sum())

        def is_legal(cx: float, cy: float) -> bool:
            if cx < half_w - 1e-9 or cx > cw - half_w + 1e-9: return False
            if cy < half_h - 1e-9 or cy > ch - half_h + 1e-9: return False
            dxg = np.abs(cx - ph_pos[:, 0])
            dyg = np.abs(cy - ph_pos[:, 1])
            return not bool(((dxg < tx) & (dyg < ty)).any())

        if is_legal(target_x, target_y):
            return target_x, target_y

        step = max(min(mw, mh) * 0.05, gap * 2.0, min(cw, ch) / 1024.0)
        best_cx, best_cy = target_x, target_y
        best_overlap = overlap_area(target_x, target_y)
        for r_step in range(1, self.radial_steps + 1):
            radius = step * r_step
            for a in range(self.radial_angles):
                theta = 2.0 * math.pi * a / self.radial_angles
                cx = float(np.clip(target_x + radius * math.cos(theta), half_w, cw - half_w))
                cy = float(np.clip(target_y + radius * math.sin(theta), half_h, ch - half_h))
                if is_legal(cx, cy):
                    return cx, cy
                ov = overlap_area(cx, cy)
                if ov < best_overlap:
                    best_overlap = ov; best_cx, best_cy = cx, cy
        return best_cx, best_cy

    # ─────────────────────────────────────────────────────────────── training

    def _train(
        self,
        benchmark: Benchmark,
        order: List[int],
        net_meta: List[Tuple[np.ndarray, np.ndarray]],
    ) -> Tuple[Optional[np.ndarray], float]:
        """Run PPO on this benchmark for up to `train_loops` loops or until the
        wall-clock budget is exhausted. Returns (best_positions[:n_hard], best_hpwl).
        """
        env = _PlaceEnv(
            benchmark, order, net_meta, grid=self.train_grid_size, gap=self.overlap_gap,
        )
        agent = _Agent(
            num_steps=env.n_steps, grid=self.train_grid_size, device=self.device,
            lr_actor=self.lr_actor, lr_critic=self.lr_critic,
            actor_lr_anneal=self.actor_lr_anneal, critic_lr_anneal=self.critic_lr_anneal,
            clip_epsilon=self.clip_epsilon, entropy_coef=self.entropy_coef,
            max_grad_norm=self.max_grad_norm, gamma=self.gamma,
        )

        best_hpwl = math.inf
        best_positions: Optional[np.ndarray] = None
        t_start = time.time()

        for loop in range(self.train_loops):
            if time.time() - t_start > self.train_time_budget:
                if self.verbose:
                    print(f"  [train] budget exhausted after loop {loop}")
                break
            buffer = _ReplayBuffer(
                capacity=self.episodes_per_loop * env.n_steps,
                grid=self.train_grid_size,
            )
            for _ep in range(self.episodes_per_loop):
                hpwl, ep_positions = self._run_episode(env, agent, buffer)
                if hpwl < best_hpwl:
                    best_hpwl = hpwl
                    best_positions = ep_positions.copy()
                if time.time() - t_start > self.train_time_budget:
                    break
            stats = agent.update(buffer, self.update_epochs, self.batch_size)
            if self.verbose:
                print(
                    f"  [train] loop {loop} best_hpwl={best_hpwl:.2f} "
                    f"actor={stats['actor_loss']:.4f} critic={stats['critic_loss']:.4f} "
                    f"entropy={stats['entropy']:.4f}"
                )

        return best_positions, best_hpwl

    def _run_episode(
        self, env: _PlaceEnv, agent: _Agent, buffer: _ReplayBuffer
    ) -> Tuple[float, np.ndarray]:
        s = env.reset()
        total_hpwl = 0.0
        t = 0
        while True:
            # Guard against the actor seeing a fully-blocked position_mask.
            if s[2].min() >= 0.5:
                action = self._fallback_action(env)
                a_logp = 0.0
            else:
                action, a_logp = agent.act(s, t, deterministic=False)
            s_next, reward, done, info = env.step(action)
            buffer.store(s, t, action, a_logp, reward, 1.0 if done else 0.0)
            total_hpwl += info["delta_hpwl"]
            s = s_next
            t += 1
            if done:
                break
        return total_hpwl, env.positions_np[:env.n_hard].copy()

    @staticmethod
    def _fallback_action(env: _PlaceEnv) -> int:
        # Pick the lowest wire_mask cell ignoring position_mask (env will clip).
        wm = env._cur_state[1].numpy()
        flat = wm.argmin()
        return int(flat)

    # ──────────────────────────────────────────── trained-policy refinement

    def _refine_with_trained(
        self,
        benchmark: Benchmark,
        full_order: List[int],
        net_meta: List[Tuple[np.ndarray, np.ndarray]],
        trained_positions: np.ndarray,
        trained_idx: List[int],
        rng: np.random.Generator,
    ) -> Tuple[np.ndarray, float]:
        """Lock in trained-policy positions for `trained_idx`, then run greedy
        on the placement grid for the remaining macros in `full_order`.
        """
        n_hard = benchmark.num_hard_macros
        trained_set = set(trained_idx)
        # Seed positions: keep init for non-trained, override with trained results.
        seed_positions = benchmark.macro_positions[:n_hard].numpy().astype(np.float64).copy()
        for i in trained_idx:
            seed_positions[i] = trained_positions[i]

        remainder = [i for i in full_order if i not in trained_set]
        # Treat trained macros as fixed by anchoring them up front.
        return self._run_greedy(
            benchmark, remainder, net_meta, rng,
            grid=self.grid_size,
            preplaced_positions=seed_positions,
            preplaced_indices=list(trained_idx),
        )

    # ─────────────────────────────────────────────────────── greedy placement

    def _run_greedy(
        self,
        benchmark: Benchmark,
        order: List[int],
        net_meta: List[Tuple[np.ndarray, np.ndarray]],
        rng: np.random.Generator,
        grid: Optional[int] = None,
        preplaced_positions: Optional[np.ndarray] = None,
        preplaced_indices: Optional[List[int]] = None,
    ) -> Tuple[np.ndarray, float]:
        n_hard = benchmark.num_hard_macros
        n_total = benchmark.num_macros
        num_ports = int(benchmark.port_positions.shape[0])
        cw = float(benchmark.canvas_width)
        ch = float(benchmark.canvas_height)
        G = grid if grid is not None else self.grid_size

        sizes_np = benchmark.macro_sizes.numpy().astype(np.float64)
        positions_np = (
            preplaced_positions.copy()
            if preplaced_positions is not None and preplaced_positions.shape[0] >= n_hard
            else benchmark.macro_positions.numpy().astype(np.float64).copy()
        )
        # Re-attach soft macros + ports if positions came in only for hard slice.
        if positions_np.shape[0] < n_total:
            full = benchmark.macro_positions.numpy().astype(np.float64).copy()
            full[:positions_np.shape[0]] = positions_np
            positions_np = full
        port_positions_np = (
            benchmark.port_positions.numpy().astype(np.float64)
            if num_ports > 0 else np.zeros((0, 2), dtype=np.float64)
        )
        fixed_np = benchmark.macro_fixed.numpy()

        gx_centers = (np.arange(G) + 0.5) * (cw / G)
        gy_centers = (np.arange(G) + 0.5) * (ch / G)

        anchored = np.zeros(n_total + num_ports, dtype=bool)
        anchored[n_hard:n_total] = True
        anchored[n_total:] = True
        for i in range(n_hard):
            if fixed_np[i]:
                anchored[i] = True
        if preplaced_indices:
            for i in preplaced_indices:
                anchored[i] = True

        n_nets = benchmark.num_nets
        net_min = np.full((n_nets, 2), np.inf, dtype=np.float64)
        net_max = np.full((n_nets, 2), -np.inf, dtype=np.float64)
        net_count = np.zeros(n_nets, dtype=np.int64)

        owner_to_nets: List[List[int]] = [[] for _ in range(n_total + num_ports)]
        owner_pins_on_net: Dict[Tuple[int, int], List[np.ndarray]] = {}
        for net_id, (owners, offsets) in enumerate(net_meta):
            for e in range(owners.shape[0]):
                o = int(owners[e])
                if 0 <= o < n_total + num_ports:
                    owner_to_nets[o].append(net_id)
                    owner_pins_on_net.setdefault((o, net_id), []).append(offsets[e])

        def owner_center(o: int) -> np.ndarray:
            return positions_np[o] if o < n_total else port_positions_np[o - n_total]

        for net_id, (owners, offsets) in enumerate(net_meta):
            for e in range(owners.shape[0]):
                o = int(owners[e])
                if not anchored[o]:
                    continue
                p = owner_center(o) + offsets[e]
                if net_count[net_id] == 0:
                    net_min[net_id] = p; net_max[net_id] = p
                else:
                    if p[0] < net_min[net_id, 0]: net_min[net_id, 0] = p[0]
                    if p[1] < net_min[net_id, 1]: net_min[net_id, 1] = p[1]
                    if p[0] > net_max[net_id, 0]: net_max[net_id, 0] = p[0]
                    if p[1] > net_max[net_id, 1]: net_max[net_id, 1] = p[1]
                net_count[net_id] += 1

        gap = self.overlap_gap
        placed_hards: List[int] = [i for i in range(n_hard) if anchored[i]]

        for macro_idx in order:
            if anchored[macro_idx]:
                continue
            mw = sizes_np[macro_idx, 0]; mh = sizes_np[macro_idx, 1]
            half_w = mw / 2.0; half_h = mh / 2.0

            valid_x = (gx_centers >= half_w - 1e-9) & (gx_centers <= cw - half_w + 1e-9)
            valid_y = (gy_centers >= half_h - 1e-9) & (gy_centers <= ch - half_h + 1e-9)
            canvas_ok = valid_x[:, None] & valid_y[None, :]

            if placed_hards:
                ph_pos = positions_np[placed_hards]
                ph_size = sizes_np[placed_hards]
                tx = (mw + ph_size[:, 0]) / 2.0 + gap
                ty = (mh + ph_size[:, 1]) / 2.0 + gap
                dx = np.abs(gx_centers[:, None] - ph_pos[None, :, 0])
                dy = np.abs(gy_centers[:, None] - ph_pos[None, :, 1])
                x_block = (dx < tx[None, :]).astype(np.int32)
                y_block = (dy < ty[None, :]).astype(np.int32)
                pos_mask = (x_block @ y_block.T) > 0
            else:
                pos_mask = np.zeros((G, G), dtype=bool)

            invalid = pos_mask | (~canvas_ok)
            wire_mask = np.zeros((G, G), dtype=np.float64)
            for net_id in owner_to_nets[macro_idx]:
                if net_count[net_id] == 0:
                    continue
                bmin0, bmin1 = net_min[net_id, 0], net_min[net_id, 1]
                bmax0, bmax1 = net_max[net_id, 0], net_max[net_id, 1]
                for off in owner_pins_on_net[(macro_idx, net_id)]:
                    px = gx_centers + off[0]
                    py = gy_centers + off[1]
                    dx2 = np.maximum(px - bmax0, 0.0) + np.maximum(bmin0 - px, 0.0)
                    dy2 = np.maximum(py - bmax1, 0.0) + np.maximum(bmin1 - py, 0.0)
                    wire_mask += dx2[:, None] + dy2[None, :]

            wm = np.where(invalid, np.inf, wire_mask)
            if not np.isfinite(wm).any():
                fallback = (~pos_mask) & canvas_ok
                if fallback.any():
                    flat = np.flatnonzero(fallback.ravel())
                    pick = int(rng.choice(flat))
                    gx, gy = divmod(pick, G)
                    cx = float(gx_centers[gx]); cy = float(gy_centers[gy])
                else:
                    target_x = float(np.clip(positions_np[macro_idx, 0], half_w, cw - half_w))
                    target_y = float(np.clip(positions_np[macro_idx, 1], half_h, ch - half_h))
                    cx, cy = self._radial_search(
                        target_x, target_y, mw, mh, half_w, half_h, cw, ch,
                        placed_hards, positions_np, sizes_np, gap,
                    )
            else:
                min_val = wm.min()
                cands = np.flatnonzero(wm.ravel() == min_val)
                pick = int(rng.choice(cands))
                gx, gy = divmod(pick, G)
                cx = float(np.clip(gx_centers[gx], half_w, cw - half_w))
                cy = float(np.clip(gy_centers[gy], half_h, ch - half_h))

            positions_np[macro_idx, 0] = cx
            positions_np[macro_idx, 1] = cy
            anchored[macro_idx] = True
            placed_hards.append(macro_idx)

            for net_id in owner_to_nets[macro_idx]:
                for off in owner_pins_on_net[(macro_idx, net_id)]:
                    p0 = cx + off[0]; p1 = cy + off[1]
                    if net_count[net_id] == 0:
                        net_min[net_id, 0] = p0; net_min[net_id, 1] = p1
                        net_max[net_id, 0] = p0; net_max[net_id, 1] = p1
                    else:
                        if p0 < net_min[net_id, 0]: net_min[net_id, 0] = p0
                        if p1 < net_min[net_id, 1]: net_min[net_id, 1] = p1
                        if p0 > net_max[net_id, 0]: net_max[net_id, 0] = p0
                        if p1 > net_max[net_id, 1]: net_max[net_id, 1] = p1
                    net_count[net_id] += 1

        multi = net_count > 1
        hpwl = float(
            (net_max[multi, 0] - net_min[multi, 0]).sum()
            + (net_max[multi, 1] - net_min[multi, 1]).sum()
        )
        return positions_np[:n_hard].copy(), hpwl
