import cv2
import numpy as np
import os
import glob
import matplotlib.pyplot as plt

# trzeba będzie usunąć rzeczy z cv2
def gabor_filter(img_gray, ksize=16, sigma=4.0, theta_step=4, freq=0.1):

    img = img_gray.astype(np.float32)
    result = np.zeros_like(img)

    for i in range(theta_step):
        theta = np.pi * i / theta_step
        kernel = cv2.getGaborKernel(
            (ksize, ksize), sigma, theta, 1.0 / freq, 0.5, 0, ktype=cv2.CV_32F
        )
        filtered = cv2.filter2D(img, cv2.CV_32F, kernel)
        np.maximum(result, filtered, out=result)

    result -= result.min()
    mx = result.max()
    if mx > 0:
        result = result / mx * 255.0

    return result.astype(np.uint8)


def morphological_close(img_binary, kernel_size=3, iterations=1):

    kernel = np.ones((kernel_size, kernel_size), dtype=np.float32)
    n = kernel_size * kernel_size

    result = img_binary.astype(np.float32)

    for _ in range(iterations):
        dilated = cv2.filter2D(result, cv2.CV_32F, kernel)
        dilated = np.where(dilated > 0, 255.0, 0.0).astype(np.float32)

        eroded = cv2.filter2D(dilated, cv2.CV_32F, kernel)
        result = np.where(eroded >= n * 255.0, 255.0, 0.0).astype(np.float32)

    return result.astype(np.uint8)


def morphological_skeletonize(img_binary):
    size = np.size(img_binary)
    skel = np.zeros(img_binary.shape, np.uint8)


    element = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
    done = False

    img_temp = img_binary.copy()

    while not done:
        eroded = cv2.erode(img_temp, element)
        temp = cv2.dilate(eroded, element)
        temp = cv2.subtract(img_temp, temp)
        skel = cv2.bitwise_or(skel, temp)
        img_temp = eroded.copy()

        zeros = size - cv2.countNonZero(img_temp)
        if zeros == size:
            done = True

    return skel


def process_fingerprints(folder_path, results_path):
    search_path = os.path.join(folder_path, '*.*')
    files = glob.glob(search_path)

    valid_extensions = ('.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff')
    files = [f for f in files if f.lower().endswith(valid_extensions)]

    if not files:
        print(f"Nie znaleziono żadnych zdjęć w folderze: {folder_path}")
        return

    print(f"Znaleziono {len(files)} plików. Rozpoczynam przetwarzanie...")

    os.makedirs(results_path, exist_ok=True)

    for file in files:
        filename = os.path.basename(file)

        img = cv2.imread(file, cv2.IMREAD_GRAYSCALE)
        if img is None:
            continue

        _, img_binary = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)

        skel_morph = morphological_skeletonize(img_binary)

        try:
            skel_zhang = cv2.ximgproc.thinning(img_binary, thinningType=cv2.ximgproc.THINNING_ZHANGSUEN)
        except AttributeError:
            skel_zhang = np.zeros_like(img_binary)
            print("Brak modułu cv2.ximgproc. Zainstaluj 'opencv-contrib-python' aby zobaczyć metodę Zhang-Suen.")


        plt.figure(figsize=(15, 5))

        plt.subplot(1, 4, 1)
        plt.imshow(img, cmap='gray')
        plt.title('Oryginał')
        plt.axis('off')

        plt.subplot(1, 4, 2)
        plt.imshow(img_binary, cmap='gray')
        plt.title('Bineryzacja')
        plt.axis('off')

        plt.subplot(1, 4, 3)
        plt.imshow(skel_morph, cmap='gray')
        plt.title('Szkieletyzacja Morfologiczna')
        plt.axis('off')

        plt.subplot(1, 4, 4)
        plt.imshow(skel_zhang, cmap='gray')
        plt.title('Ścienianie Zhang-Suen')
        plt.axis('off')

        plt.suptitle(f"Plik: {filename}", fontsize=16)
        plt.tight_layout()

        name_without_ext = os.path.splitext(filename)[0]

        save_path = os.path.join(results_path, f"result_{name_without_ext}.png")
        plt.savefig(save_path)
        plt.close()

        print(f"Zapisano wynik dla: {filename}")


if __name__ == "__main__":
    # Ścieżka do folderu ze zdjęciami oraz folderu wynikowego
    input_folder = "Proj3/RODO"
    results_folder = "results"

    # Tworzenie folderu wejściowego w przypadku, gdyby nie istniał
    if not os.path.exists(input_folder):
        print(f"Folder {input_folder} nie istnieje. Utwórz folder i dodaj do niego obrazy.")
    else:
        process_fingerprints(input_folder, results_folder)
        print("Przetwarzanie zakończone.")