import tkinter as tk
from tkinter import filedialog, messagebox, ttk
import numpy as np
import cv2
from PIL import Image, ImageTk

from k3m import k3m
from Proj3 import morphological_skeletonize, gabor_filter, morphological_close


def np_to_tk(arr, max_size=280):
    if len(arr.shape) == 3:
        img = Image.fromarray(cv2.cvtColor(arr, cv2.COLOR_BGR2RGB))
    else:
        arr_ui = arr.astype(np.uint8) if arr.dtype != np.uint8 else arr
        img = Image.fromarray(arr_ui)
    img.thumbnail((max_size, max_size), Image.LANCZOS)
    return ImageTk.PhotoImage(img)


def get_minutiae_data(skel_img):
    binary = (skel_img > 0).astype(np.uint8)
    h, w = binary.shape
    out_img = cv2.cvtColor(skel_img, cv2.COLOR_GRAY2BGR)
    points = []

    y, x = np.where(binary == 1)
    if len(x) > 0:

        points_for_hull = np.column_stack((x, y))
        hull = cv2.convexHull(points_for_hull)

        blob_mask = np.zeros((h, w), dtype=np.uint8)
        cv2.fillConvexPoly(blob_mask, hull, 1)

        kernel_erode = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (41, 41))
        safe_zone = cv2.erode(blob_mask, kernel_erode)
    else:
        safe_zone = np.zeros((h, w), dtype=np.uint8)

    for r in range(1, h - 1):
        for c in range(1, w - 1):


            if binary[r, c] == 1 and safe_zone[r, c] == 1:
                p = [binary[r - 1, c - 1], binary[r - 1, c], binary[r - 1, c + 1],
                     binary[r, c + 1], binary[r + 1, c + 1], binary[r + 1, c],
                     binary[r + 1, c - 1], binary[r, c - 1]]

                cn = 0
                for i in range(8):
                    cn += abs(int(p[i]) - int(p[(i + 1) % 8]))
                cn = cn // 2

                if cn == 1:
                    cv2.circle(out_img, (c, r), 3, (0, 0, 255), 1)
                    points.append((r, c, 1))
                elif cn == 3:
                    cv2.circle(out_img, (c, r), 3, (255, 0, 0), 1)
                    points.append((r, c, 3))

    return out_img, points


def match_minutiae(points1, points2, tolerance=2):
    matches = 0
    used_idx2 = set()
    for r1, c1, t1 in points1:
        for i, (r2, c2, t2) in enumerate(points2):
            if i not in used_idx2 and t1 == t2:
                dist = np.sqrt((r1 - r2)**2 + (c1 - c2)**2)
                if dist <= tolerance:
                    matches += 1
                    used_idx2.add(i)
                    break
    return matches


class BiometriaApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Analiza Odcisków Palców - Projekt 3")
        self.filepath = None
        self.file_compare = [None, None]
        self.photo_refs = []

        top = tk.Frame(root)
        top.pack(side=tk.TOP, pady=5)
        tk.Button(top, text="Wczytaj obraz", command=self.load_file).pack(side=tk.LEFT, padx=5)
        self.label_path = tk.Label(top, text="Nie wybrano pliku")
        self.label_path.pack(side=tk.LEFT, padx=5)
        tk.Button(top, text="Przetwórz wszystko", command=self.process).pack(side=tk.LEFT, padx=5)

        self.notebook = ttk.Notebook(root)
        self.notebook.pack(fill=tk.BOTH, expand=True)

        self.tab1 = tk.Frame(self.notebook)
        self.notebook.add(self.tab1, text="Etapy przetwarzania")
        self.img_labels_t1 = []
        labels_t1 = ["Oryginał", "Binaryzacja", "Szkielet morfologiczny", "K3M", "K3M + Gabor + Close"]
        for i, txt in enumerate(labels_t1):
            f = tk.Frame(self.tab1);
            f.grid(row=0, column=i, padx=8, pady=10)
            tk.Label(f, text=txt, font=("Arial", 9, "bold")).pack()
            l = tk.Label(f);
            l.pack();
            self.img_labels_t1.append(l)

        self.tab2 = tk.Frame(self.notebook)
        self.notebook.add(self.tab2, text="Porównanie Minucji")
        self.img_labels_t2 = []
        labels_t2 = ["Minucje: Szkielet Morfologiczny", "Minucje: K3M + Gabor + Close"]
        for i, txt in enumerate(labels_t2):
            f = tk.Frame(self.tab2);
            f.grid(row=0, column=i, padx=20, pady=10)
            tk.Label(f, text=txt, font=("Arial", 10, "bold")).pack()
            l = tk.Label(f);
            l.pack();
            self.img_labels_t2.append(l)

        self.tab3 = tk.Frame(self.notebook)
        self.notebook.add(self.tab3, text="Weryfikacja")
        f_up = tk.Frame(self.tab3);
        f_up.pack(pady=10)
        tk.Button(f_up, text="Odcisk A", command=lambda: self.load_for_comp(0)).grid(row=0, column=0, padx=5)
        tk.Button(f_up, text="Odcisk B", command=lambda: self.load_for_comp(1)).grid(row=0, column=1, padx=5)
        tk.Button(self.tab3, text="Porównaj", command=self.compare_fingerprints).pack(pady=5)
        self.result_text = tk.Label(self.tab3, text="Wynik: -", font=("Arial", 12, "bold"))
        self.result_text.pack(pady=10)
        f_res = tk.Frame(self.tab3);
        f_res.pack()
        self.comp_img_l = tk.Label(f_res);
        self.comp_img_l.pack(side=tk.LEFT, padx=10)
        self.comp_img_r = tk.Label(f_res);
        self.comp_img_r.pack(side=tk.LEFT, padx=10)

        root.geometry("1500x650")

    def load_file(self):
        path = filedialog.askopenfilename()
        if path:
            self.filepath = path
            self.label_path.config(text=path)

    def load_for_comp(self, idx):
        path = filedialog.askopenfilename()
        if path: self.file_compare[idx] = path

    def process(self):
        if not self.filepath: return
        img_gray = cv2.imread(self.filepath, cv2.IMREAD_GRAYSCALE)

        _, img_bin = cv2.threshold(img_gray, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)

        skel_morph = morphological_skeletonize(img_bin)
        skel_k3m_raw = k3m(img_bin)

        img_gabor = gabor_filter(img_gray)
        _, img_bin_gabor = cv2.threshold(img_gabor, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)
        img_clean = morphological_close(cv2.morphologyEx(img_bin_gabor, cv2.MORPH_OPEN, np.ones((3, 3))), 3)
        skel_k3m_best = k3m(img_clean)

        img_min_morph, _ = get_minutiae_data(skel_morph)
        img_min_k3m, _ = get_minutiae_data(skel_k3m_best)

        self.photo_refs = [
            np_to_tk(img_gray), np_to_tk(img_bin), np_to_tk(skel_morph),
            np_to_tk(skel_k3m_raw), np_to_tk(skel_k3m_best),
            np_to_tk(img_min_morph), np_to_tk(img_min_k3m)
        ]

        for i in range(5): self.img_labels_t1[i].config(image=self.photo_refs[i])
        for i in range(2): self.img_labels_t2[i].config(image=self.photo_refs[5 + i])

    def compare_fingerprints(self):
        if not all(self.file_compare): return
        pts_data = []
        imgs_ui = []
        for path in self.file_compare:
            img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
            img_g = gabor_filter(img)
            _, img_b = cv2.threshold(img_g, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)
            img_c = morphological_close(cv2.morphologyEx(img_b, cv2.MORPH_OPEN, np.ones((3, 3))), 3)
            skel = k3m(img_c)

            res_img, pts = get_minutiae_data(skel)

            pts_data.append(pts)
            imgs_ui.append(np_to_tk(res_img, max_size=300))

        m_count = match_minutiae(pts_data[0], pts_data[1])
        match = m_count >= 15
        self.result_text.config(
            text=f"Wynik: {'To ten sam odcisk' if match else 'Różne odciski'} (Wspólne: {m_count})",
            fg="green" if match else "red"
        )
        self.comp_img_l.config(image=imgs_ui[0])
        self.comp_img_r.config(image=imgs_ui[1])
        self.photo_refs.extend(imgs_ui)


if __name__ == "__main__":
    root = tk.Tk()
    app = BiometriaApp(root)
    root.mainloop()