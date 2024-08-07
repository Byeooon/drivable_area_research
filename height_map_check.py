import cv2
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

# img = cv2.imread('/home/julio981007/HDD/orfd/validation/height/1620326361639.png', cv2.IMREAD_GRAYSCALE).astype(np.float32)
img = cv2.imread('/home/julio981007/HDD/orfd/validation/height/1620326361639.tiff', cv2.IMREAD_UNCHANGED)

plt.imshow(img)
plt.colorbar()
plt.show()