import numpy as np
import matplotlib.pyplot as plt
import sys

# Controllo argomenti
if len(sys.argv) < 3:
    print("Uso: python3 audio_neural_net_3D_advanced.py input.wav output.jpeg")
    sys.exit(1)

input_file = sys.argv[1]
output_file = sys.argv[2]

# Leggi byte audio
with open(input_file, "rb") as f:
    data = f.read()
arr = np.frombuffer(data, dtype=np.uint8)
arr = arr[44:] if len(arr) > 100 else arr  # salta header WAV

# Riduci lunghezza per velocità
max_nodes = 1000
total_bytes = len(arr)
if total_bytes < 3:
    print("❌ Audio troppo corto!")
    sys.exit(1)

# Prendi pacchetti di 3 byte
num_packets = min(max_nodes, total_bytes // 3)
arr = arr[:num_packets*3].reshape((num_packets,3))

# Nodo 3D: ogni pacchetto → (x, y, z)
x = arr[:,0] / 255.0
y = arr[:,1] / 255.0
z = arr[:,2] / 255.0

# Colore RGB dai byte
colors = arr / 255.0

# Dimensione nodo dal valore medio del pacchetto
sizes = np.mean(arr, axis=1) / 2 + 10

# Crea figura ad alta risoluzione
fig = plt.figure(figsize=(20,20), dpi=100)
ax = fig.add_subplot(111, projection='3d')
ax.axis('off')

# Disegna connessioni tra nodi vicini
threshold = 0.15  # distanza massima
for i in range(num_packets):
    for j in range(i+1, num_packets):
        dist = np.sqrt((x[i]-x[j])**2 + (y[i]-y[j])**2 + (z[i]-z[j])**2)
        if dist < threshold:
            ax.plot([x[i], x[j]], [y[i], y[j]], [z[i], z[j]],
                    color=(colors[i,0], colors[i,1], colors[i,2]),
                    alpha=0.3, linewidth=1)

# Disegna nodi
ax.scatter(x, y, z, s=sizes, c=colors, edgecolors='w', alpha=0.9)

# Salva come JPEG 2000x2000
plt.savefig(output_file, dpi=100, bbox_inches='tight')
plt.close()

print(f"✅ Rete neurale 3D avanzata salvata come {output_file} a 2000x2000 px")
