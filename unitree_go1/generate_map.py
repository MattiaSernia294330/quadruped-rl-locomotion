import numpy as np
from PIL import Image

# Parametri della heightmap
N = 1025   # dimensione heightmap
K = 40     # numero di picchi montuosi

# Coordinate normalizzate da -1 a 1
x = np.linspace(-1, 1, N)
y = np.linspace(-1, 1, N)
X, Y = np.meshgrid(x, y)
R = np.sqrt(X**2 + Y**2)

# Base a forma di valle al centro (inverted Gaussian)
valley_sigma = 0.3
base = 1 - np.exp(- (R**2) / (2 * valley_sigma**2))
base = (base - base.min()) / (base.max() - base.min())

# Aggiunta di montagne sparse e non regolari, solo fuori dal cerchio centrale di raggio 0.3
mountains = np.zeros_like(base)
np.random.seed(123)
count = 0
while count < K:
    # Posizione casuale
    cx = np.random.uniform(-1, 1)
    cy = np.random.uniform(-1, 1)
    if np.sqrt(cx**2 + cy**2) < 0.3:
        continue
    amplitude = np.random.uniform(0.05, 0.15)   # ampiezza picco più piccola
    sigma = np.random.uniform(0.02, 0.05)       # dimensione picco compatta
    dist = np.sqrt((X - cx)**2 + (Y - cy)**2)
    mountains += amplitude * np.exp(-(dist**2) / (2 * sigma**2))
    count += 1

# Composizione finale: valle + montagne
height = base + mountains
height = (height - height.min()) / (height.max() - height.min())  # normalizzazione

# Salvataggio in PNG 8-bit
img = (height * 255).astype(np.uint8)
Image.fromarray(img).save('mountain_heightmap.png')

