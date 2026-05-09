import tkinter as tk
from tkinter import filedialog, messagebox
import numpy as np
import cv2
from PIL import Image, ImageTk

from k3m import k3m

from Proj3 import morphological_skeletonize

def np_to_tk(arr, max_size=300):
    if arr.dtype != np.uint8:
        arr = arr.astype(np.uint8)
    img = Image.fromarray(arr)
    img.thumbnail((max_size, max_size), Image.LANCZOS)
    return ImageTk.PhotoImage(img)


class BiometriaApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Porównanie algorytmów ścieniania")

        self.filepath = None
        self.photo_refs = []  # zapobiegamy garbage collection

        # --- Górny pasek: wybór pliku i przycisk ---
        top = tk.Frame(root)
        top.pack(side=tk.TOP, pady=4)

        tk.Button(top, text="Wczytaj plik", command=self.load_file).pack(side=tk.LEFT, padx=4)
        self.label_path = tk.Label(top, text="Nie wybrano pliku")
        self.label_path.pack(side=tk.LEFT, padx=4)
        tk.Button(top, text="Przetwórz", command=self.process).pack(side=tk.LEFT, padx=4)

        # --- Obszar wyników: 4 kolumny ---
        results = tk.Frame(root)
        results.pack(side=tk.TOP, pady=4)

        labels = ["Oryginał", "Binaryzacja", "Szkielet morfologiczny", "K3M"]
        self.img_labels = []
        self.caption_labels = []

        for i, title in enumerate(labels):
            col = tk.Frame(results)
            col.grid(row=0, column=i, padx=8)
            tk.Label(col, text=title).pack()
            img_lbl = tk.Label(col)
            img_lbl.pack()
            self.img_labels.append(img_lbl)

        root.geometry("1400x600")

    def load_file(self):
        path = filedialog.askopenfilename(
            filetypes=[("Obrazy", "*.png *.jpg *.jpeg *.bmp *.tif *.tiff")]
        )
        if path:
            self.filepath = path
            self.label_path.config(text=path)

    def process(self):
        if not self.filepath:
            messagebox.showwarning("Brak pliku", "Najpierw wczytaj plik.")
            return

        img_gray = cv2.imread(self.filepath, cv2.IMREAD_GRAYSCALE)
        if img_gray is None:
            messagebox.showerror("Błąd", "Nie można wczytać obrazu.")
            return

        _, img_bin = cv2.threshold(
            img_gray, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU
        )

        skel_morph = morphological_skeletonize(img_bin)
        skel_k3m = k3m(img_bin)

        images = [img_gray, img_bin, skel_morph, skel_k3m]

        self.photo_refs.clear()
        for lbl, arr in zip(self.img_labels, images):
            ph = np_to_tk(arr)
            self.photo_refs.append(ph)
            lbl.config(image=ph)


if __name__ == "__main__":
    root = tk.Tk()
    app = BiometriaApp(root)
    root.mainloop()
