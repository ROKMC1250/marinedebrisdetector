import rasterio as rio
import numpy as np
from PIL import Image
L1CBANDS = ["B1", "B2", "B3", "B4", "B5", "B6", "B7", "B8", "B8A", "B9", "B10", "B11", "B12"]
L2ABANDS = ["B1", "B2", "B3", "B4", "B5", "B6", "B7", "B8", "B8A", "B9", "B11", "B12"]
def read_tif_image(imagefile, window=None):
    # loading of the image
    with rio.open(imagefile, "r") as src:
        image = src.read(window=window)

        is_l1cimage = src.meta["count"] == 13  # flag if l1c (top-of-atm) or l2a (bottom of atmosphere) image

        # keep only 12 bands: delete 10th band (nb: 9 because start idx=0)
        if is_l1cimage:  # is L1C Sentinel 2 data
            image = image[[L1CBANDS.index(b) for b in L2ABANDS]]

        if window is not None:
            win_transform = src.window_transform(window)
        else:
            win_transform = src.transform
    return image, win_transform

image = read_tif_image('datasets/MARIDA/scenes/S2_SR_20201015T152651_20201015T152645_T18QYF.tif')
rgb_image = np.stack((image[0][0], image[0][1], image[0][2]), axis=-1)
print(rgb_image[0].max())
print(rgb_image[1].max())
print(rgb_image[2].max())
# print(max(rgb_image[1]))
# print(max(rgb_image[2]))
# Pillow 이미지 객체로 변환
image_ = Image.fromarray(rgb_image.astype(np.uint8))

# PNG 파일로 저장
image_.save("output.png")
# print(image[0].shape)