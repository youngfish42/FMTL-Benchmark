import os
from concurrent.futures import ProcessPoolExecutor
from shutil import copyfile

import matplotlib.pyplot as plt
from PIL import Image


def pick(root_folder, sub_folders, out_folder, file_name):
    if not os.path.exists(out_folder):
        os.makedirs(out_folder)

    rows = len(root_folder)
    cols = len(sub_folders[0]) + 1
    fig, ax = plt.subplots(rows, cols, figsize=(cols * 2, rows * 2))

    for i in range(rows):
        for j in range(cols):
            if j == 0:
                if "PASCALContext" in root_folder[0]:
                    img_name = os.path.join(root_folder[0], "JPEGImages", file_name + ".jpg")
                elif "NYUDv2" in root_folder[0]:
                    img_name = os.path.join(root_folder[0], "images", file_name + ".png")
            else:
                folder = root_folder[i]
                img_name = os.path.join(folder, sub_folders[(i > 0)][j - 1], file_name + ".png")

            img = Image.open(img_name)
            ax[i, j].imshow(img)
            ax[i, j].axis("off")
        ax[i, 4].axis("off")

    plt.subplots_adjust(top=1, bottom=0, right=1, left=0, hspace=0, wspace=0)
    plt.margins(0, 0)
    plt.savefig(os.path.join(out_folder, file_name + ".png"), dpi=300)


def collect(root_folder, sub_folders, prefix, suffix, out_root, file_name):
    out_folder = os.path.join(out_root, file_name)
    if not os.path.exists(out_folder):
        os.makedirs(out_folder)

    # copyfile(os.path.join(root_folder[0], "JPEGImages", file_name + ".jpg"),
    #          os.path.join(out_folder, file_name + ".jpg"))
    copyfile(os.path.join(root_folder[0], "images", file_name + ".png"), os.path.join(out_folder, file_name + ".png"))

    for i in range(len(root_folder)):
        if i == 0:
            sf = sub_folders[0]
        else:
            sf = sub_folders[1]
        for j in range(len(sf)):
            copyfile(os.path.join(root_folder[i], sf[j], file_name + ".png"),
                     os.path.join(out_folder, prefix[i] + '_' + file_name + suffix[j] + ".png"))


if __name__ == "__main__":
    # first is dataset
    # PASCAL
    # root_folder = [
    #     "/data/datasets/PASCALContext", "exp/local_new3/predictions", "exp/fedavg/predictions",
    #     "exp/manytask/predictions", "exp/ab_c04/predictions"
    # ]
    # sub_folders = [["semseg/pascal-context", "parts_vis", "sal_distill", "normals_distill", "edge_vis"],
    #                ["0_semseg", "1_human_parts", "4_sal", "2_normals", "3_edge/img"]]
    # list_file = "ImageSets/Context/val.txt"

    # NYUD
    root_folder = [
        "/data/datasets/NYUDv2", "exp/local_new3/predictions", "exp/fedavg/predictions", "exp/manytask/predictions",
        "exp/ab_c04/predictions"
    ]
    sub_folders = [["semseg_vis", "depth_vis", "normals", "edge"], ["5_semseg", "5_depth", "5_normals", "5_edge/img"]]
    # list_file = "gt_sets/val.txt"

    ### pick ###
    # with open(os.path.join(root_folder[0], list_file), "r") as f:
    #     file_names = [x.strip() for x in f.readlines()]

    # file_names = os.listdir(os.path.join(root_folder[0], "parts_vis"))
    # file_names = file_names[0:2]
    # print(len(file_names))
    # file_names = ["2008_000034.jpg"]

    # with ProcessPoolExecutor(max_workers=48) as executor:
    #     for file_name in file_names:
    #         file_name = file_name.split(".")[0]
    #         executor.submit(pick, root_folder, sub_folders, "pick_ny/", file_name)

    # for file_name in file_names:
    #     file_name = file_name.split(".")[0]
    #     pick(root_folder, sub_folders, "pick2/", file_name)

    ### collect ###
    prefix = ["gt", "local", "avg", "mat", "hca2"]
    # suffix = ["ss", "p", "sal", "n", "e"]
    suffix = ["ss", "d", "n", "e"]
    collect(root_folder, sub_folders, prefix, suffix, "collect", "0446")

    print("Done")
