import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import morphology
from skimage.filters import threshold_otsu, threshold_triangle
from skimage.measure import label, regionprops


def circularity(region, label=1):
    return (region.perimeter**2) / region.area


def count_lakes(image):
    B = ~image
    BB = np.ones((B.shape[0] + 2, B.shape[1] + 2))
    BB[1:-1, 1:-1] = B
    return (np.max(label(BB)) - 1)


def has_vline(image):
    lines = np.sum(image, 0) // image.shape[0]
    return 1 in lines


def has_bay(image):
    b = ~image
    bb = np.zeros((b.shape[0] + 1, b.shape[1])).astype("uint8")
    bb[:-1, :] = b
    return count_lakes(~bb)


def count_bays(image):
    holes = ~image.copy()
    return np.max(label(holes))


def recognize(region):
    lakes = count_lakes(region.image)

    if (lakes == 0):
        bays = count_bays(region.image)

        if (bays == 0):
            return '-'
        elif (bays == 2):
            return '/'
        elif (bays == 3):
            isvert = has_vline(region.image)

            if (isvert):
                return '1'
        elif (bays == 4):
            isbay = has_bay(region.image)

            if (isbay == 1):
                return 'X'
            else:
                return '*'
        elif (bays == 5):
            circ = circularity(region)

            if (circ < 50):
                return '*'
            else:
                return 'W'
    elif (lakes == 1):
        bays = count_bays(region.image)

        if (bays == 3):
            circ = circularity(region)

            if (circ > 59):
                return 'D'
            else:
                return 'P'
        elif (bays == 4):
            return 'A'
        elif (bays == 5):
            return '0'
    elif (lakes == 2):
        bays = count_bays(region.image)

        if (bays <= 4):
            return 'B'
        elif (bays >= 6):
            return '8'

    return None


image = plt.imread(
    "C:\\symbols\\symbols.png")
image = np.sum(image, 2)
image[image > 0] = 1

d = {}
labeled = label(image)
regions = regionprops(labeled)
print(np.max(labeled))

fig, axarr = plt.subplots(10, 40)
fig.patch.set_facecolor('black')
plt.set_cmap('gray')

for (i, region) in enumerate(regions):
    axarr[i // 40, i % 40].imshow(region.image)
    axarr[i // 40, i % 40].axis('off')

    symbol = recognize(region)

    title = axarr[i // 40, i % 40].set_title(label=symbol)
    plt.setp(title, color='cyan')

    if symbol not in d:
        d[symbol] = 1
    else:
        d[symbol] += 1
print(d)

plt.show()