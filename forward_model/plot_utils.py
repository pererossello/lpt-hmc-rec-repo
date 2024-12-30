import sys

import numpy as np
import jax.numpy as jnp
import matplotlib.pyplot as plt
from matplotlib.ticker import FixedLocator

from .stats import get_pdf, get_pow_spec_1D, get_cross_power_spec_1D, get_reduced_bispectrum


def plot_cubes(
    list_of_arrays,
    cmap="seismic_r",
    figsize=5,
    axis=0,
    idx=0,
    width=1,
    vlim=None,
    wspace=0.05,
    return_ims=False,
):

    if not isinstance(list_of_arrays, list):
        list_of_arrays = [list_of_arrays]
    M = len(list_of_arrays)

    if not isinstance(cmap, list):
        cmap = [cmap] * M

    if vlim is not None:
        if isinstance(vlim, (float, int)):
            vlim = [(-vlim, vlim)] * M
        elif isinstance(vlim, list):
            for i, vlim_item in enumerate(vlim):
                if isinstance(vlim_item, (int, float)):
                    vlim[i] = (-vlim_item, vlim_item)
        elif isinstance(vlim, tuple):
            vlim = [(vlim[0], vlim[1])] * M

    if M < 4:
        n_col = M
        n_row = 1
    elif M == 4:
        n_col = 2
        n_row = 2

    ratio = n_col / n_row
    figsize_fact = figsize / np.sqrt(ratio)
    fig, axs = plt.subplots(n_row, n_col, figsize=(figsize_fact * ratio, figsize_fact))
    plt.subplots_adjust(wspace=wspace, hspace=0.05)

    ims = []
    axs_flat = axs.flatten() if M > 1 else [axs]
    for i, ax in enumerate(axs_flat):
        ax.set_xticks([])
        ax.set_yticks([])

        arr = list_of_arrays[i]
        arr, min_val, max_val = get_projection(arr, axis, idx, width)

        vmin = min_val if vlim is None else vlim[i][0]
        vmax = max_val if vlim is None else vlim[i][1]

        im = ax.imshow(arr.T, vmin=vmin, vmax=vmax, origin="lower", cmap=cmap[i])
        ims.append(im)

    if return_ims:
        return fig, axs, ims

    return fig, axs

def get_projection(array, axis, idx, width):

    N = array.shape[0]

    if idx > N - 1:
        raise ValueError("idx should be smaller than N-1")

    idx_min = idx
    idx_max = idx + width

    if axis == 0:
        matrix = array[idx_min:idx_max, :, :]
    elif axis == 1:
        matrix = array[:, idx_min:idx_max, :]
    elif axis == 2:
        matrix = array[:, :, idx_min:idx_max]

    matrix = np.sum(matrix, axis=axis) / width
    min_val = np.min(matrix)
    max_val = np.max(matrix)
    # abs_max_val = np.max([np.abs(min_val), np.abs(max_val)])

    return matrix, min_val, max_val



def compare_pow_spec(delta_list, L,  n_bins=50, labels=None, title=None, xlog=False, no_labels=False, sphere_only=False, lw=0.5, x_lim=None):

    M = len(delta_list)

    if labels == None:
        labels = [f'{i}' for i in range(M)]

    
    ls = ['-'] + ['--']*(M-1)
    cmap = plt.get_cmap('turbo') 
    cs = [cmap(i) for i in np.linspace(0.1, 0.9, M-1, endpoint=False)]
    cs = ['k'] + cs 

    figsize = 5
    lw = lw*figsize

    labelsize = 11
    fig, axs = plt.subplots(1, 3, figsize=(3*figsize, figsize))
    plt.subplots_adjust(wspace=0.15, hspace=None)

    for i, delta in enumerate(delta_list):
        k,pk = get_pow_spec_1D(delta, L, n_bins=n_bins, sphere_only=sphere_only)

        

        axs[0].plot(k, pk, c=cs[i], ls=ls[i], label=labels[i], lw=lw)


        if i==0:
            pk0 = pk.copy()
        else:
            axs[1].plot(k, pk/pk0, c=cs[i], ls=ls[i], lw=lw)

            k, cross = get_cross_power_spec_1D(delta_list[0], delta_list[i], L, n_bins, sphere_only=sphere_only)
            axs[2].plot(k, cross, c=cs[i], ls=ls[i], lw=lw)



    axs[0].set_yscale('log')

    axs[2].set_ylim(-0.2, 1.1)
    if x_lim is not None:
        axs[0].set_xlim()
    #axs[0].set_xscale('log')

    if not no_labels:
        axs[0].legend(fontsize=labelsize)


    axs[0].tick_params(labelsize=labelsize, top=True)

    #axs[1].set_ylim(0.5, 2)
    axs[1].tick_params(labelsize=labelsize, left=False, labelleft=False, right=True, labelright=True, top=True)

    axs[2].tick_params(labelsize=labelsize, left=False, labelleft=False, right=True, labelright=True, top=True)

    for ax in axs:

        ax.grid(which="major",
            linewidth=0.5,
            color="k",
            alpha=0.25,)
        ax.set_xlabel('$k$ [$h$/Mpc]', fontsize=labelsize)
        

        if xlog:
            ax.set_xscale('log')
        else:
            ax.set_xlim(0,  None)
            xticks = ax.get_xticks()
            ax2 = ax.twiny()
            ax2.set_xlim(axs[0].get_xlim())  # Ensure the limits of top and 
            ax2.xaxis.set_major_locator(FixedLocator(xticks))
            ax2.set_xticklabels(['']+[f"{2*np.pi/label:0.2f}" for label in xticks[1:]])
            ax2.set_xlabel(r'$\lambda$ [Mpc/$h$]', fontsize=labelsize, labelpad=7)
            ax2.tick_params(labelsize=labelsize)



    if title is not None:
        fig.suptitle(title, y=1.05, fontsize=labelsize*1.4)

    return fig, axs


def compare_deltas(
    delta_ref, delta_list, L, n_pdf_bins=100, n_pk_bins=50, n_thetas=100, ref_label='ref', labels=None
):

    if not isinstance(delta_list, list):
        delta_list = [delta_list]

    M = len(delta_list)
    if labels == None:
        labels = [""] * M

    # COLORS
    c_ref = "k"
    cmap = plt.get_cmap("turbo")
    cs = [cmap(i) for i in np.linspace(0.1, 0.9, M, endpoint=False)]

    # BISPEC CONFIG
    thetas = jnp.linspace(0, jnp.pi, n_thetas)
    k1, k2 = 0.1, 0.2

    fs, ratio = 3.5, 4
    fig, axs = plt.subplots(1, 4, figsize=(fs * ratio, fs))

    # SET AUX PANELS
    height = 0.2
    axs_ = []
    for ax in axs[:3]:
        ax.tick_params(labelbottom=False)
        ax_pos = ax.get_position()
        ax_ = fig.add_axes(
            [
                ax_pos.x0,  # x position
                ax_pos.y0-height,  # y position
                ax_pos.width,  # width
                height,  # height
            ]
        )
        axs_.append(ax_)




    # PDF STUFF
    all_max = (delta_ref + 1.0).max()
    all_min = (delta_ref + 1.0).min()

    pdf_bin_edges = jnp.linspace(all_min, all_max, n_pdf_bins + 1)
    pdf_bin_centers = pdf_bin_edges[:-1] + jnp.diff(pdf_bin_edges) * 0.5

    pdf_ref = get_pdf(delta_ref + 1.0, pdf_bin_edges)

    axs[0].plot(pdf_bin_centers, pdf_ref, c=c_ref, label=ref_label)

    for i, delta in enumerate(delta_list):
        pdf = get_pdf(delta + 1.0, pdf_bin_edges)
        axs[0].plot(pdf_bin_centers, pdf, c=cs[i], ls="--", label=labels[i])

        axs_[0].plot(pdf_bin_centers, pdf/pdf_ref, ls="--", label=labels[i], c=cs[i])

    # PK STUFF
    k_bin_centers, pk_ref = get_pow_spec_1D(delta_ref, L, n_pk_bins)
    axs[1].plot(k_bin_centers, pk_ref, c=c_ref)
    for i, delta in enumerate(delta_list):
        _, pk = get_pow_spec_1D(delta, L, n_pk_bins)
        axs[1].plot(k_bin_centers, pk, c=cs[i], ls="--")

        axs_[1].plot(k_bin_centers, pk/pk_ref, ls="--", label=labels[i], c=cs[i])

    # BISPECTRA
    bispec_ref = get_reduced_bispectrum(delta_ref, L, k1, k2, thetas)
    axs[2].plot(thetas, bispec_ref, c=c_ref)
    for i, delta in enumerate(delta_list):
        bispec = get_reduced_bispectrum(delta, L, k1, k2, thetas)
        axs[2].plot(thetas, bispec, c=cs[i], ls="--")

        axs_[2].plot(thetas, bispec/bispec_ref, ls="--", label=labels[i], c=cs[i])

    # CROSS
    for i, delta in enumerate(delta_list):
        k_bin_centers, pk_cross = get_cross_power_spec_1D(delta_ref, delta, L, n_bins=n_pk_bins)
        axs[3].plot(k_bin_centers, pk_cross, c=cs[i], ls="--")


    axs[0].set_title('PDF')
    axs[0].legend()
    axs[0].set_yscale("log")
    axs[0].set_xscale("log")
    axs_[0].set_xscale("log")

    axs[1].set_title('Power Spectrum')
    axs[1].set_yscale("log")
    axs[1].set_xscale("log")
    axs_[1].set_xscale("log")

    axs_[1].set_ylim(0.25, 1.25)

    axs[2].set_title('Reduced Bispectrum')

    axs[3].set_title('Cross Power Spectrum')
    axs[3].set_xscale('log')
    axs[3].set_ylim(0,1.01)

    for ax in list(axs)+axs_:
        ax.grid(
            which="major",
            linewidth=0.5,
            color="k",
            alpha=0.25,
        )

    return fig
    


def scatter_density(ax, x, y, step=1, n_bins=250, cmap='turbo', s=1., x_lim=None, y_lim=None):

    x = x.ravel()[::step]
    y = y.ravel()[::step]

    if x_lim is None:
        min_x, max_x = np.min(x), np.max(x)
    else: 
        min_x, max_x = x_lim
    if y_lim is None:
        min_y, max_y = np.min(y), np.max(y)
    else:
        min_y, max_y = y_lim
    
    hist, x_edges, y_edges = np.histogram2d(
        x, y, bins=n_bins, 
        range=[[min_x, max_x], [min_y, max_y]]
    )

    edges1 = np.linspace(min_x, max_x, n_bins)
    edges2 = np.linspace(min_y, max_y, n_bins)

    delta_edge1 = np.diff(edges1)[0]
    idxs1 = np.int32((x-min_x)/delta_edge1)

    delta_edge2 = np.diff(edges2)[0]
    idxs2 = np.int32((y-min_y)/delta_edge2)

    hist = hist[idxs1, idxs2]
    hist = hist/np.max(hist)

    ax.scatter(x, y, s=s, lw=0., c=hist, cmap=cmap)