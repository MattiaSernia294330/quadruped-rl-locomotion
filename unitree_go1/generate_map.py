import numpy as np
from PIL import Image

# Parametri della heightmap
N = 1025  # dimensione molto grande: 1025x1025
K = 30    # numero di picchi montuosi casuali

# Coordinate normalizzate da -1 a 1
x = np.linspace(-1, 1, N)
y = np.linspace(-1, 1, N)
X, Y = np.meshgrid(x, y)
R = np.sqrt(X**2 + Y**2)

# Base a forma di valle centrale (più bassa al centro, più alta sui bordi)
base = np.clip(R, 0, 1)

# Aggiunta di montagne sparse:
mountains = np.zeros_like(base)
np.random.seed(42)  # per riproducibilità
for _ in range(K):
    # posizione casuale lungo i bordi della mappa
    angle = np.random.rand() * 2 * np.pi
    radius = 0.7 + 0.3 * np.random.rand()  # verso il bordo esterno
    cx, cy = radius * np.cos(angle), radius * np.sin(angle)
    
    # ampiezza e dimensione del picco
    amplitude = 0.2 + 0.3 * np.random.rand()
    sigma = 0.05 + 0.1 * np.random.rand()
    
    # gaussiana
    dist = np.sqrt((X - cx)**2 + (Y - cy)**2)
    mountains += amplitude * np.exp(-(dist**2) / (2 * sigma**2))

# Composizione finale: valle + montagne
height = base + mountains

# Normalizzazione su [0,1]
height = (height - height.min()) / (height.max() - height.min())

# Conversione in immagine 8-bit e salvataggio
img = (height * 255).astype(np.uint8)
Image.fromarray(img).save('mountain_heightmap.png')

