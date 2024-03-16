import logging

import matplotlib.pyplot as plt

from data.build_dataset import LunaDataset
from data.load_data import Ct

logging.getLogger("PIL").setLevel(logging.WARNING)
logging.getLogger("matplotlib").setLevel(logging.WARNING)


def find_positive_samples(luna_dataset, limit=100):
    positive_sample_list = []
    for sample_tup in luna_dataset.candidateInfo_list:
        if sample_tup.isNodule_bool:
            print(len(positive_sample_list), sample_tup)
            positive_sample_list.append(sample_tup)
        if len(positive_sample_list) >= limit:
            break
    return positive_sample_list


def show_candidate(series_uid, batch_ndx=None, **kwargs):
    ds = LunaDataset(series_uid=series_uid, **kwargs)
    clim = (-1000.0, 300)
    pos_list = [i for i, x in enumerate(ds.candidateInfo_list) if x.isNodule_bool]

    if batch_ndx is None:
        if pos_list:
            batch_ndx = pos_list[0]
        else:
            print("Warning: no positive samples found; using first negative sample.")
            batch_ndx = 0

    ct = Ct(series_uid)
    ct_t, pos_t, series_uid, center_irc = ds[batch_ndx]
    ct_a = ct_t[0].numpy()

    fig = plt.figure(figsize=(10, 20))

    group_list = [
        [9, 11, 13],
        [15, 16, 17],
        [19, 21, 23],
    ]
    font_size = 10

    subplot = fig.add_subplot(len(group_list) + 2, 3, 1)
    subplot.set_title('index {}'.format(int(center_irc[0])), fontsize=font_size)
    for label in (subplot.get_xticklabels() + subplot.get_yticklabels()):
        label.set_fontsize(font_size)
    plt.imshow(ct.hu_a[int(center_irc[0])], clim=clim, cmap='gray')

    subplot = fig.add_subplot(len(group_list) + 2, 3, 2)
    subplot.set_title('row {}'.format(int(center_irc[1])), fontsize=font_size)
    for label in (subplot.get_xticklabels() + subplot.get_yticklabels()):
        label.set_fontsize(font_size)
    plt.imshow(ct.hu_a[:, int(center_irc[1])], clim=clim, cmap='gray')
    plt.gca().invert_yaxis()

    subplot = fig.add_subplot(len(group_list) + 2, 3, 3)
    subplot.set_title('col {}'.format(int(center_irc[2])), fontsize=font_size)
    for label in (subplot.get_xticklabels() + subplot.get_yticklabels()):
        label.set_fontsize(font_size)
    plt.imshow(ct.hu_a[:, :, int(center_irc[2])], clim=clim, cmap='gray')
    plt.gca().invert_yaxis()

    subplot = fig.add_subplot(len(group_list) + 2, 3, 4)
    subplot.set_title('index {}'.format(int(center_irc[0])), fontsize=font_size)
    for label in (subplot.get_xticklabels() + subplot.get_yticklabels()):
        label.set_fontsize(font_size)
    plt.imshow(ct_a[ct_a.shape[0] // 2], clim=clim, cmap='gray')

    subplot = fig.add_subplot(len(group_list) + 2, 3, 5)
    subplot.set_title('row {}'.format(int(center_irc[1])), fontsize=font_size)
    for label in (subplot.get_xticklabels() + subplot.get_yticklabels()):
        label.set_fontsize(font_size)
    plt.imshow(ct_a[:, ct_a.shape[1] // 2], clim=clim, cmap='gray')
    plt.gca().invert_yaxis()

    subplot = fig.add_subplot(len(group_list) + 2, 3, 6)
    subplot.set_title('col {}'.format(int(center_irc[2])), fontsize=font_size)
    for label in (subplot.get_xticklabels() + subplot.get_yticklabels()):
        label.set_fontsize(font_size)
    plt.imshow(ct_a[:, :, ct_a.shape[2] // 2], clim=clim, cmap='gray')
    plt.gca().invert_yaxis()

    for row, index_list in enumerate(group_list):
        for col, index in enumerate(index_list):
            subplot = fig.add_subplot(len(group_list) + 2, 3, row * 3 + col + 7)
            subplot.set_title('slice {}'.format(index), fontsize=font_size)
            for label in (subplot.get_xticklabels() + subplot.get_yticklabels()):
                label.set_fontsize(font_size)
            plt.imshow(ct_a[index], clim=clim, cmap='gray')
    plt.show()
    # print(series_uid, batch_ndx, bool(pos_t[0]), pos_list)
