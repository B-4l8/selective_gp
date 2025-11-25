#!/usr/bin/env python

import torch
import numpy as np
import matplotlib.pyplot as plt


def plot_samples(model, ax=None, resolution=1000, n_samples=10, xlim=None):
    if ax is None:
        ax = plt.subplots()[1]

    if xlim is None:
        xlim = ax.get_xlim()

    x_in = torch.linspace(*xlim, resolution)
    y_samples = []
    for _ in range(n_samples):
        f = x_in
        for gp in model.gps:
            f_dist = gp(f[:, None])
            f_dist._covar = f_dist._covar.add_jitter()
            f = f_dist.sample()
        y_samples.append(f.tolist())

    ax.plot(x_in, np.transpose(y_samples), color=(0, 0, 0, 0.1))


def plot_density(model, ax, resolution=10, n_samples=50, xlim=None, ylim=None,
                 cmap=plt.cm.Blues):

    if xlim is None:
        xlim = ax.get_xlim()

    if ylim is None:
        ylim = ax.get_ylim()

    xmin, xmax = xlim
    ymin, ymax = ylim

    def _add_density(x_array, y_array):
        i = ((y_array - ymin) / (ymax - ymin) * dim_y).to(int)
        j = ((x_array - xmin) / (xmax - xmin) * dim_x).to(int)
        for i_, j_ in zip(i, j):
            try:
                M[i_, j_] += 1.0
            except IndexError:
                continue

    dim_x = int((xmax - xmin) * n_samples)
    dim_y = int((ymax - ymin) * n_samples)
    M = np.zeros((dim_y, dim_x))

    x_in = torch.linspace(xmin, xmax, dim_x)
    for _ in range(n_samples):
        f = x_in
        for gp in model.gps:
            f_dist = gp(f[:, None])
            m, s = f_dist.mean, f_dist.stddev
            f = torch.distributions.Normal(loc=m, scale=s).sample()
        _add_density(x_in, f)

    yr = np.linspace(ymin, ymax, dim_y)
    xr = np.linspace(xmin, xmax, dim_x)
    y, x = np.meshgrid(xr, yr)
    ax.pcolormesh(y, x, M, cmap=cmap, zorder=-1, vmin=0, vmax=M.max())
    ax.set(xlim=(xmin, xmax), ylim=(ymin, ymax))


def plot_deep_latent(model, axes=None, xlims=None, **kwargs):
    if axes is None:
        axes = plt.subplots(1, len(model.gps))[1]
        for ax in axes:
            ax.margins(x=0)

    if xlims is None:
        xlims = [None] * len(model.gps)
    else:
        assert len(xlims) == len(model.gps)

    for gp, ax, xlim in zip(model.gps, axes, xlims):
        plot_latent(gp, ax=ax, xlim=xlim, **kwargs)


# def plot_latent(gp, ax=None, xlim=None, resolution=200, cmap=plt.cm.Set1):
#     if ax is None:
#         ax = plt.subplots()[1]

#     with torch.no_grad():
#         x_u = gp.inducing_inputs.clone().flatten()
#         u = gp.inducing_distribution.mean.clone()

#         p = gp.variational_point_process.probabilities
#         color = [(0, 0, 0, p_) for p_ in p]

#     ax.scatter(x_u, u, color=color, edgecolor="k")

#     if xlim is None:
#         xlim = ax.get_xlim()

#     x_ = torch.linspace(*xlim, resolution)

#     with torch.no_grad():
#         f_dist = gp(x_[:, None])

#     m, s = f_dist.mean, f_dist.stddev
#     ax.plot(x_, m, color=cmap(1))
#     ax.fill_between(x_, m - s, m + s, color=cmap(1, 0.3))
def plot_latent(gp, ax=None, xlim=None, resolution=200, cmap=plt.cm.Set1):
    if ax is None:
        ax = plt.subplots()[1]

    device = next(gp.parameters()).device # device 확인

    with torch.no_grad():
        x_u = gp.inducing_inputs.clone().flatten().cpu().numpy()
        u = gp.inducing_distribution.mean.clone().cpu().numpy()

        p = gp.variational_point_process.probabilities.cpu().numpy()
        color = [(0, 0, 0, p_) for p_ in p]

    ax.scatter(x_u, u, color=color, edgecolor="k")

    if xlim is None:
        xlim = ax.get_xlim()

    x_ = torch.linspace(*xlim, resolution).to(device)

    with torch.no_grad():
        f_dist = gp(x_[:, None])

    m = f_dist.mean.cpu().numpy()
    s = f_dist.stddev.cpu().numpy()
    x_cpu = x_.cpu().numpy()

    ax.plot(x_cpu, m, color=cmap(1))
    ax.fill_between(x_cpu, m - s, m + s, color=cmap(1, 0.3))

def plot_latent_2d(gp, ax=None, xlim=None, ylim=None, resolution=50, cmap="RdBu_r"):
    """
    2차원 GP 모델의 잠재 함수(Latent Function)와 유도 포인트(Inducing Points)를 그립니다.
    """
    if ax is None:
        ax = plt.gca()

    device = gp.inducing_inputs.device

    # 1. 유도 포인트(Inducing Points) 그리기
    # 2차원 좌표(x1, x2)를 추출합니다.
    with torch.no_grad():
        X_u = gp.inducing_inputs.clone().cpu().numpy()
        
        # 가지치기(Pruning) 확률 가져오기
        if hasattr(gp, 'variational_point_process'):
            p = gp.variational_point_process.probabilities.cpu().numpy()
            # 투명도(Alpha)를 확률에 따라 조절 (검은색)
            # 확률이 낮을수록 투명해져서 사라지는 것처럼 보임
            colors = [(0, 0, 0, float(p_)) for p_ in p]
        else:
            colors = 'black'

    # 배경(Contour) 위에 점을 그리기 위해 나중에 scatter를 호출하는 것이 일반적이지만,
    # 사용자 코드 순서를 고려해 일단 데이터 준비만 합니다.

    # 2. 격자(Grid) 생성 및 예측 (결정 경계 그리기)
    if xlim is None:
        xlim = ax.get_xlim()
    if ylim is None:
        ylim = ax.get_ylim()
        
    # 축 범위가 설정되지 않았을 경우를 대비한 안전장치
    if xlim[0] == xlim[1]: xlim = (-2, 12)
    if ylim[0] == ylim[1]: ylim = (-2, 12)

    xs = np.linspace(xlim[0], xlim[1], resolution)
    ys = np.linspace(ylim[0], ylim[1], resolution)
    xx, yy = np.meshgrid(xs, ys)
    
    # 모델 입력용 텐서 생성 (N, 2)
    grid_tensor = torch.tensor(np.column_stack([xx.ravel(), yy.ravel()]), dtype=torch.float64, device=device)
    
    with torch.no_grad():
        # GP 잠재 함수값 예측 (f_mean)
        # 이진 분류에서 f_mean > 0 이면 클래스 1, < 0 이면 클래스 0 입니다.
        f_dist = gp(grid_tensor)
        pred_mean = f_dist.mean.cpu().numpy().reshape(xx.shape)
        # pred_std = f_dist.stddev.cpu().numpy().reshape(xx.shape) # 분산이 필요하면 사용

    # 등고선(Contour) 그리기: 모델의 결정 경계
    # 0을 기준으로 파란색(양수)과 빨간색(음수)으로 나뉩니다.
    ax.contourf(xx, yy, pred_mean, levels=20, cmap=cmap, alpha=0.4)
    
    # 결정 경계선 (f=0) 추가 (점선)
    ax.contour(xx, yy, pred_mean, levels=[0], colors='k', linestyles='--', linewidths=1)

    # 유도 포인트 산점도 그리기
    # marker='D' (다이아몬드), 흰색 테두리로 눈에 잘 띄게 처리
    ax.scatter(X_u[:, 0], X_u[:, 1], c=colors, s=60, marker='D', edgecolors='white', linewidth=0.8, label="Inducing Points")


def plot_probabilities(gp, ax=None, color=None):
    if ax is None:
        ax = plt.subplots()[1]

    with torch.no_grad():
        x_u = gp.inducing_inputs.clone().flatten()
        p = gp.variational_point_process.probabilities

    if color is None:
        color = plt.cm.Set(0)[:3]
    ax.bar(x_u, p, color=color + (0.5,), width=2, edgecolor=color)
    ax.set_ylim(0, 1)
