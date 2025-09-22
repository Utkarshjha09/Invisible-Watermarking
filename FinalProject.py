import cv2
import numpy as np
import tkinter as tk
from tkinter import filedialog, messagebox


def embed_watermark(image_path, watermark_path, save_path):
    img = cv2.imread(image_path)
    watermark = cv2.imread(watermark_path, cv2.IMREAD_GRAYSCALE)

    img_float = np.float32(img) / 255.0
    ycrcb = cv2.cvtColor(img_float, cv2.COLOR_BGR2YCrCb)
    y_channel = ycrcb[:, :, 0]
    dct = cv2.dct(y_channel)

    h, w = dct.shape
    watermark_resized = cv2.resize(watermark, (w, h), interpolation=cv2.INTER_AREA)
    watermark_norm = np.float32(watermark_resized) / 255.0
    alpha = 0.05
    dct_watermarked = dct + alpha * watermark_norm

    y_watermarked = cv2.idct(dct_watermarked)
    y_watermarked = np.clip(y_watermarked, 0, 1)
    ycrcb[:, :, 0] = y_watermarked
    watermarked_img = cv2.cvtColor(ycrcb, cv2.COLOR_YCrCb2BGR)
    watermarked_img_uint8 = np.uint8(watermarked_img * 255)

    cv2.imwrite(save_path, watermarked_img_uint8)
    messagebox.showinfo("Success", f"Watermarked image saved at:\n{save_path}")


def extract_watermark(original_image_path, watermarked_image_path, save_path):
    original_img = cv2.imread(original_image_path)
    watermarked_img = cv2.imread(watermarked_image_path)

    original_float = np.float32(original_img) / 255.0
    watermarked_float = np.float32(watermarked_img) / 255.0

    original_ycrcb = cv2.cvtColor(original_float, cv2.COLOR_BGR2YCrCb)
    watermarked_ycrcb = cv2.cvtColor(watermarked_float, cv2.COLOR_BGR2YCrCb)

    y_original = original_ycrcb[:, :, 0]
    y_watermarked = watermarked_ycrcb[:, :, 0]

    dct_original = cv2.dct(y_original)
    dct_watermarked = cv2.dct(y_watermarked)

    alpha = 0.05
    diff_dct = (dct_watermarked - dct_original) / alpha
    extracted_watermark = np.clip(diff_dct, 0, 1)
    extracted_watermark_uint8 = np.uint8(extracted_watermark * 255)

    cv2.imwrite(save_path, extracted_watermark_uint8)
    messagebox.showinfo("Success", f"Extracted watermark saved at:\n{save_path}")


# ---------------- GUI Setup ----------------

def embed_gui():
    embed_window = tk.Toplevel(root)
    embed_window.title("Embed Watermark")

    tk.Label(embed_window, text="Select Image to Watermark:").pack(pady=5)
    img_path_var = tk.StringVar()
    tk.Entry(embed_window, textvariable=img_path_var, width=50).pack()
    tk.Button(embed_window, text="Browse", command=lambda: browse_file(img_path_var)).pack()

    tk.Label(embed_window, text="Select Watermark Image:").pack(pady=5)
    watermark_path_var = tk.StringVar()
    tk.Entry(embed_window, textvariable=watermark_path_var, width=50).pack()
    tk.Button(embed_window, text="Browse", command=lambda: browse_file(watermark_path_var)).pack()

    tk.Button(embed_window, text="Save As", command=lambda: save_watermarked(img_path_var.get(), watermark_path_var.get())).pack(pady=20)


def browse_file(path_var):
    file_path = filedialog.askopenfilename(filetypes=[("Image files", "*.jpg *.jpeg *.png")])
    if file_path:
        path_var.set(file_path)


def save_watermarked(image_path, watermark_path):
    if not image_path or not watermark_path:
        messagebox.showerror("Error", "Please select both files!")
        return
    save_path = filedialog.asksaveasfilename(defaultextension=".png",
                                             filetypes=[("PNG files", "*.png"), ("JPEG files", "*.jpg")])
    if save_path:
        embed_watermark(image_path, watermark_path, save_path)


def extract_gui():
    extract_window = tk.Toplevel(root)
    extract_window.title("Extract Watermark")

    tk.Label(extract_window, text="Select Original Image:").pack(pady=5)
    original_path_var = tk.StringVar()
    tk.Entry(extract_window, textvariable=original_path_var, width=50).pack()
    tk.Button(extract_window, text="Browse", command=lambda: browse_file(original_path_var)).pack()

    tk.Label(extract_window, text="Select Watermarked Image:").pack(pady=5)
    watermarked_path_var = tk.StringVar()
    tk.Entry(extract_window, textvariable=watermarked_path_var, width=50).pack()
    tk.Button(extract_window, text="Browse", command=lambda: browse_file(watermarked_path_var)).pack()

    tk.Button(extract_window, text="Save Extracted Watermark", command=lambda: save_extracted(original_path_var.get(), watermarked_path_var.get())).pack(pady=20)


def save_extracted(original_image_path, watermarked_image_path):
    if not original_image_path or not watermarked_image_path:
        messagebox.showerror("Error", "Please select both files!")
        return
    save_path = filedialog.asksaveasfilename(defaultextension=".png",
                                             filetypes=[("PNG files", "*.png"), ("JPEG files", "*.jpg")])
    if save_path:
        extract_watermark(original_image_path, watermarked_image_path, save_path)


# ---------------- Main Window ----------------

root = tk.Tk()
root.title("DCT Watermarking Tool")
root.geometry("1200x900")

tk.Label(root, text="Choose an operation", font=("Arial", 14)).pack(pady=20)

tk.Button(root, text="Embed Watermark", width=20, command=embed_gui).pack(pady=10)
tk.Button(root, text="Extract Watermark", width=20, command=extract_gui).pack(pady=10)
tk.Button(root, text="Exit", width=20, command=root.quit).pack(pady=10)

root.mainloop()
