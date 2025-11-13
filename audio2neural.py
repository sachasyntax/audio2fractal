import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import sys

if len(sys.argv) < 3:
    print("Uso: python3 audio2neural.py input.wav output.jpeg")
    sys.exit(1)

input_file = sys.argv[1]
output_file = sys.argv[2]

with open(input_file, "rb") as f:
    data = f.read()
arr = np.frombuffer(data, dtype=np.uint8)
arr = arr[44:] if len(arr) > 100 else arr  

max_nodes = 800
if len(arr) > max_nodes:
    arr = arr[::len(arr)//max_nodes]
num_nodes = len(arr)

np.random.seed(42)
x = np.random.rand(num_nodes)
y = np.random.rand(num_nodes)

colors = arr / 255
sizes = arr / 2 + 10  # dimensione minima 10

fig, ax = plt.subplots(figsize=(20,20), dpi=100)  
ax.axis('off')

threshold = 0.2
for i in range(num_nodes):
    for j in range(i+1, num_nodes):
        dist = np.sqrt((x[i]-x[j])**2 + (y[i]-y[j])**2)
        if dist < threshold:
            ax.plot([x[i], x[j]], [y[i], y[j]],
                    color=(colors[i], colors[j], 1-colors[i]),
                    alpha=0.3, linewidth=1)

ax.scatter(x, y, s=sizes, c=colors, cmap='plasma', edgecolors='w', alpha=0.9)

plt.savefig(output_file, dpi=100, bbox_inches='tight')
plt.close()

print(f"image file saved as {output_file}")
