from skimage.metrics import structural_similarity as calculate_ssim
from tqdm import tqdm
import cv2
import os
import random
from threading import Thread, Lock
import itertools
from glob import glob
from utils import config
from utils.generic_utils import print_log


def split(a, n):
    k, m = divmod(len(a), n)
    return list(a[i * k + min(i, m):(i + 1) * k + min(i + 1, m)] for i in range(n))


def resize(h1, h2, mode='avg'):
    if mode == 'avg':
        size0 = int((h1.shape[0] + h2.shape[0]) / 2)
        size1 = int((h1.shape[1] + h2.shape[1]) / 2)

    elif mode == 'min':
        size0 = min(h1.shape[0], h2.shape[0])
        size1 = min(h1.shape[1], h2.shape[1])

    else:  # mode == 'max'
        size0 = max(h1.shape[0], h2.shape[0])
        size1 = max(h1.shape[1], h2.shape[1])

    return cv2.resize(h1, (size1, size0)), cv2.resize(h2, (size1, size0))


def sub_ifssim(combs, lock, avg_fam):
    for (h1, h2) in combs:
        heatmap1, heatmap2 = cv2.imread(h1), cv2.imread(h2)
        if heatmap1.shape != heatmap2.shape:
            heatmap1, heatmap2 = resize(heatmap1, heatmap2, mode='avg')
        ssim = calculate_ssim(heatmap1, heatmap2, multichannel=True if heatmap1.shape[2] == 3 else False)
        with lock:
            avg_fam[0] += ssim


def find_path(sample, hetmap_folder, hetmaps_complete):
    path = None
    if os.path.isfile(f"{hetmap_folder}/heatmap_{sample}"):
        path = f"{hetmap_folder}/heatmap_{sample}"
    else:  # when sample was not classified correctly and has "WRONG" in sample name, use complete path for matching
        for c in hetmaps_complete:
            if sample in c:
                path = f"{hetmap_folder}/{c}"
    return path


def IFIM_SSIM(args):
    folders = args.ssim_folders
    if len(folders) == 1 and folders[0] == 'ALL':
        folders = glob(config.main_path + 'results/images/*/', recursive=True)
    elif len(folders) > 1:
        for i, f in enumerate(folders):
            if not os.path.exists(f):
                if os.path.exists(config.main_path + 'results/images/' + f):
                    folders[i] = config.main_path + 'results/images/' + f
                else:
                    print_log(f"ERROR! Folder '{f}' not found, exiting...", print_on_screen=True)
                    exit()
    else:
        print_log(f"ERROR! You need to provide, at least, 2 folders. Exiting...", print_on_screen=True)
        exit()

    print_log(f"IF-SSIM per model and class", print_on_screen=True)
    for ex in folders:
        print_log(f"---- FOLDER {ex} ----", print_on_screen=True)

        for fam in os.listdir(ex):
            avg_fam = [0]  # Counter define as a mutable object (list) because it will be persistent across threads

            available_heatmaps = [os.path.join(dirpath, f) for (dirpath, dirnames, filenames) in
                                  os.walk(f"{ex}/{fam}/heatmap") for f in filenames]

            # Check if FLAG --include_all was set, otherwise choose 50 samples to run the IF-SSIM analysis on
            if not args.include_all and len(available_heatmaps) > 50:
                available_heatmaps = random.sample(available_heatmaps, 50)

            # Calculate all combinations of lenght 2 among the available heatmaps
            combinations = list(itertools.combinations(available_heatmaps, 2))

            lock = Lock()
            threads = []

            # Split the combinations in 5 separated groups and assign the analysis to a different thread to parallelize
            for part in split(combinations, 5):
                t = Thread(target=sub_ifssim, args=[part, lock, avg_fam])
                threads.append(t)
                t.start()

            for t in threads:
                t.join()

            avg_fam = avg_fam[0] / len(combinations)

            print_log(f"{fam} = {avg_fam}", print_on_screen=True)

    print_log(f"IM-SSIM per couple of models and class", print_on_screen=True)
    models = list(itertools.combinations(folders, 2))

    for (m1, m2) in models:
        print_log(f"---- COMPARISON {m1} - {m2} ----", print_on_screen=True)

        # Use the first output class to evaluate cardinality of samples (check if this class in both models)
        fam_test = os.listdir(m1)[0]
        if fam_test not in os.listdir(m2):
            print_log(f"ERROR! Output class '{fam_test}' from {m1} not found in {m2}. Exiting...", print_on_screen=True)
            exit()

        # Count number of samples for the same class with regard to model 1 and model 2
        card_fam_test1 = len([name for name in os.listdir(f"{m1}/{fam_test}/heatmap")])
        card_fam_test2 = len([name for name in os.listdir(f"{m2}/{fam_test}/heatmap")])

        # Choose first_model the one with less samples, It will be the base model for evaluate the SSIM difference,
        #  since it has less samples, it means less iterations in the next loop
        if card_fam_test1 < card_fam_test2:
            first_model = m1
            second_model = m2
        else:
            first_model = m2
            second_model = m1

        # Iterate over all the Output classes of first model
        for fam in os.listdir(first_model):
            avg_fam = 0
            n = 0

            # Check if 'fam' output class also in second model
            if fam not in os.listdir(second_model):
                print_log(f"ERROR! Output class '{fam}' from {first_model} not found in {second_model}. Exiting...",
                          print_on_screen=True)
                exit()

            # Extract all heatmap samples
            heatmap_folder_firstmodel = f"{first_model}/{fam}/heatmap"
            heatmap_folder_secondmodel = f"{second_model}/{fam}/heatmap"
            # TODO: improve sample name extractions!
            #  So far: assuming that name format 'heatmap[WRONGcorrectClass]_samplename'
            # The first two variables store only the file name, the others the complete name (also 'WRONG', if present)
            h_firstmodel = [name.split('_')[1:][0] for name in os.listdir(heatmap_folder_firstmodel)]
            h_secondmodel = [name.split('_')[1:][0] for name in os.listdir(heatmap_folder_secondmodel)]
            h_firstmodel_complete = [name for name in os.listdir(heatmap_folder_firstmodel)]
            h_secondmodel_complete = [name for name in os.listdir(heatmap_folder_secondmodel)]

            # For each heatmap sample of first model, look for heatmap sample with same name in second model
            for i in tqdm(h_firstmodel):

                if i in h_secondmodel:  # The two models has heatmap for sample 'i'

                    path_firstmodel = find_path(i, heatmap_folder_firstmodel, h_firstmodel_complete)
                    if path_firstmodel is None:
                        print_log(f"Path for {i} in {heatmap_folder_firstmodel} not found!")
                        continue

                    path_secondmodel = find_path(i, heatmap_folder_secondmodel, h_secondmodel_complete)
                    if path_secondmodel is None:
                        print_log(f"Path for {i} in {heatmap_folder_secondmodel} not found!")
                        continue

                    h1 = cv2.imread(path_firstmodel)
                    h2 = cv2.imread(path_secondmodel)

                    if h1.shape != h2.shape:
                        h1, h2 = resize(h1, h2, mode='avg')
                    avg_fam += calculate_ssim(h1, h2,
                                              multichannel=True if h1.shape[2] == 3 else False)
                    n += 1

            avg_fam = avg_fam / n

            print_log(f"{fam} = {avg_fam} (matching {n} samples out of {len(h_firstmodel)})", print_on_screen=True)
