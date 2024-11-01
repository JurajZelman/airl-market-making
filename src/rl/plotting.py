"""Methods for plotting and monitoring."""

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.gridspec import GridSpec


def visualize_bc_train_stats(train_stats: dict):
    """
    Visualize the training statistics of the behavior cloning agent.

    Args:
        train_stats: Training statistics.
    """
    figsize = (12, 18)

    fig = plt.figure(figsize=figsize)
    gs = GridSpec(4, 2, figure=fig)

    ax1 = plt.subplot(gs[0, 0])
    ax2 = plt.subplot(gs[0, 1])
    ax3 = plt.subplot(gs[1, 0])
    ax4 = plt.subplot(gs[1, 1])
    ax5 = plt.subplot(gs[2, 0])
    ax6 = plt.subplot(gs[2, 1])
    ax7 = plt.subplot(gs[3, 0])
    # ax8 = plt.subplot(gs[3, 1])

    x = train_stats["num_samples_so_far"]

    # Loss plot
    ax1.plot(x, train_stats["loss"])
    ax1.set_title("Loss")

    # Entropy plot
    ax2.plot(x, train_stats["entropy"])
    ax2.set_title("Entropy")

    # Entropy loss plot
    ax3.plot(x, train_stats["ent_loss"])
    ax3.set_title("Entropy loss")

    # Probability of true action plot
    ax4.plot(x, train_stats["prob_true_act"])
    ax4.set_title("Probability of true action")

    # L2 loss plot
    ax5.plot(x, train_stats["l2_loss"])
    ax5.set_title(r"$L_2$ loss")

    # L2 norm plot
    ax6.plot(x, train_stats["l2_norm"])
    ax6.set_title(r"$L_2$ norm")

    # Neglogp plot
    ax7.plot(x, train_stats["neglogp"])
    ax7.set_title("Neglogp")

    plt.tight_layout()
    plt.show()


def visualize_airl_train_stats(
    train_stats: dict, save_fig: bool = False
) -> None:
    """
    Visualize the training statistics of the AIRL trainer.

    Args:
        train_stats: Training statistics.
        save_fig: Whether to save figures.
    """
    figsize_disc = (12, 18)
    figsize_gen = (12, 18)

    # --------------------------------------------------
    # Discriminator plot
    # --------------------------------------------------
    fig = plt.figure(figsize=figsize_disc)
    gs = GridSpec(4, 2, figure=fig)

    ax1 = fig.add_subplot(gs[0, :])
    ax2 = fig.add_subplot(gs[1, 0])
    ax3 = fig.add_subplot(gs[1, 1])
    ax4 = fig.add_subplot(gs[2, 0])
    ax5 = fig.add_subplot(gs[2, 1])
    ax6 = fig.add_subplot(gs[3, 0])
    ax7 = fig.add_subplot(gs[3, 1])

    # Discriminator loss
    ax1.plot(train_stats["mean/disc/disc_loss"])
    ax1.set_title("Discriminator loss")

    # Discriminator accuracy
    ax2.plot(train_stats["mean/disc/disc_acc"])
    ax2.set_title("Discriminator accuracy")

    # Discriminator entropy
    ax3.plot(train_stats["mean/disc/disc_entropy"])
    ax3.set_title("Discriminator entropy")

    # Discriminator accuracy (expert)
    ax4.plot(train_stats["mean/disc/disc_acc_expert"])
    ax4.set_title("Discriminator accuracy (expert)")

    # Discriminator accuracy (generator)
    ax5.plot(train_stats["mean/disc/disc_acc_gen"])
    ax5.set_title("Discriminator accuracy (generator)")

    # Discriminator expert proportion (true)
    ax6.plot(train_stats["mean/disc/disc_proportion_expert_true"])
    ax6.set_title("Proportion of expert actions (true)")

    # Discriminator expert proportion (predicted)
    ax7.plot(train_stats["mean/disc/disc_proportion_expert_pred"])
    ax7.set_title("Proportion of expert actions (predicted)")

    fig.tight_layout()
    if save_fig:
        fig.savefig("images/disc_train_stats.pdf")

    # --------------------------------------------------
    # Generator plot
    # --------------------------------------------------
    fig = plt.figure(figsize=figsize_gen)
    gs = GridSpec(4, 2, figure=fig)

    ax1 = fig.add_subplot(gs[0, :])
    # ax2 = fig.add_subplot(gs[0, 1])
    ax2 = fig.add_subplot(gs[1, 0])
    ax3 = fig.add_subplot(gs[1, 1])
    ax4 = fig.add_subplot(gs[2, 0])
    ax5 = fig.add_subplot(gs[2, 1])
    ax6 = fig.add_subplot(gs[3, 0])
    ax7 = fig.add_subplot(gs[3, 1])

    # Generator loss
    ax1.plot(train_stats["mean/gen/train/loss"])
    ax1.set_title("Generator loss")

    # Generator entropy loss
    ax2.plot(train_stats["mean/gen/train/entropy_loss"])
    ax2.set_title("Generator entropy loss")

    # Generator explained variance
    ax3.plot(np.clip(train_stats["mean/gen/train/explained_variance"], -1, 1))
    ax3.set_title("Generator explained variance")

    # Generator value loss
    ax4.plot(train_stats["mean/gen/train/value_loss"])
    ax4.set_title("Generator value loss")

    # Generator policy gradient loss
    ax5.plot(train_stats["mean/gen/train/policy_gradient_loss"])
    ax5.set_title("Generator policy gradient loss")

    # Generator clip fraction
    ax6.plot(train_stats["mean/gen/train/clip_fraction"])
    ax6.set_title("Generator clip fraction")

    # Generator approx kl
    ax7.plot(train_stats["mean/gen/train/approx_kl"])
    ax7.set_title("Generator approximate Kullback-Leibler div")

    fig.tight_layout()
    # fig.show()
    # Save figure
    if save_fig:
        fig.savefig("images/gen_train_stats.pdf")

    fig = plt.figure(figsize=(12, 4))
    plt.plot(train_stats["mean/gen/rollout/ep_rew_mean"])
    # Set y-axis limits
    plt.ylim(0, 310)
    plt.title("Mean episode reward")
    if save_fig:
        fig.savefig("images/mean_ep_rew.pdf")
