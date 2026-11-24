"""GMM diagnostic plots and qualitative overlay rendering."""
import cv2
import matplotlib.lines as line2d
import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as stats

try:
    import scienceplots  # noqa: F401

    plt.style.use(["science", "ieee"])
except Exception:
    pass


def plot_two_histograms(
    data1,
    data2,
    plot_name="None",
    bins=30,
    labels=("Nominal Sampels", "Anomalous Samples"),
    colors=("blue", "orange"),
    alpha=0.6,
    vline=None,
    vline2=None,
    vline_color="red",
    vline_label="Reference Line",
    gaussian_0=None,
    gaussian_1=None,
):
    combined = np.concatenate(
        [np.asarray(data1, dtype=float), np.asarray(data2, dtype=float)]
        if len(data2)
        else [np.asarray(data1, dtype=float), np.asarray(data1, dtype=float)]
    )
    bin_edges = np.histogram_bin_edges(combined, bins=bins)

    fig, ax1 = plt.subplots(1, 1)
    ax1.set_xlabel("Anomaly Score")
    ax1.set_ylabel("Occurances")
    ax1.hist(data1, bins=bin_edges, color=colors[0], alpha=alpha, label=labels[0])
    ax1.hist(data2, bins=bin_edges, color=colors[1], alpha=alpha, label=labels[1])
    if vline is not None:
        ax1.axvline(
            vline, color=vline_color, linestyle="--", linewidth=1, label="Crossover Threshold"
        )
    if vline2 is not None:
        ax1.axvline(vline2, color="black", linestyle="--", linewidth=1, label="Old Method Thresold")

    ax2 = ax1.twinx()
    ax2.set_ylabel("")
    ax2.set_axis_off()
    if gaussian_0 is not None:
        g_0 = np.linspace(
            gaussian_0["mu"] - 10 * gaussian_0["sigma"],
            gaussian_0["mu"] + 10 * gaussian_0["sigma"],
            250,
        )
        ax2.plot(
            g_0,
            stats.norm.pdf(g_0, gaussian_0["mu"], gaussian_0["sigma"]) * gaussian_0["weight"],
            color="black",
            linestyle="-",
            label="Dual Fit Gaussians",
        )
    if gaussian_1 is not None:
        g_1 = np.linspace(
            gaussian_1["mu"] - 10 * gaussian_1["sigma"],
            gaussian_1["mu"] + 10 * gaussian_1["sigma"],
            250,
        )
        ax2.plot(
            g_1,
            stats.norm.pdf(g_1, gaussian_1["mu"], gaussian_1["sigma"]) * gaussian_1["weight"],
            color="black",
            linestyle="-",
            label="Second Gaussian",
        )
    handles, _ = ax1.get_legend_handles_labels()
    handles.append(line2d.Line2D([0], [0], color="black", linestyle="-", label="Dual Fit Gaussians"))
    plt.legend(handles=handles)
    plt.title("")
    plt.xlim(0, 1)
    plt.grid(True)
    plt.savefig(plot_name + ".pdf")
    plt.clf()
    plt.close()


def visualize_confidence(img, pred, truth=None, display_size=(512, 512)):
    if not isinstance(img, np.ndarray):
        raise TypeError("img must be a numpy.ndarray")
    if not isinstance(pred, np.ndarray):
        raise TypeError("pred must be a numpy.ndarray")

    pred = cv2.resize(pred, display_size)
    if len(img.shape) == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    img = cv2.resize(img, display_size)

    center = img.shape
    x = center[1] / 2 - 224 * 2 / 2
    y = center[0] / 2 - 224 * 2 / 2
    img = img[int(y) : int(y + 224 * 2), int(x) : int(x + 224 * 2)]
    img = cv2.resize(img, display_size)

    pred = np.expand_dims(pred, 2)
    pred = np.repeat(pred, 3, axis=2)
    pred = np.clip(pred, 0, 1)

    score_img = cv2.applyColorMap(((1 - pred) * 255).astype(np.uint8), cv2.COLORMAP_HOT)
    score_img = cv2.resize(score_img, display_size)
    score_img = cv2.cvtColor(score_img, cv2.COLOR_BGR2RGB)

    pred[pred < 0.33] = ((pred[pred < 0.33] * 3) ** 4) / 3
    pred = np.clip(pred, 0, 0.9)
    img = (img.astype(np.float32) * (1 - pred) + score_img.astype(np.float32) * pred).astype(
        np.uint8
    )

    if truth is not None:
        truth = np.transpose(truth, (1, 2, 0))
        if np.max(truth) > 1:
            truth = truth / np.max(truth)
        truth = (truth * 255.0).astype(np.uint8)
        truth = cv2.resize(truth, display_size)
        edged = cv2.Canny(truth, 30, 200)
        contours, _ = cv2.findContours(edged, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(img, contours, -1, (180, 105, 255), 2)
    return img
