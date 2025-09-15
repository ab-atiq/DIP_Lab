import subprocess
import sys

for task in ['task4_quantization.py', 'task5_bitplane.py']:
    subprocess.run([sys.executable, task])
