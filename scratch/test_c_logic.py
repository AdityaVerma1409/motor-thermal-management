import json
import re

with open('mth_calibration.h', 'r', encoding='utf-8', errors='ignore') as f:
    text = f.read()

defs = {}
for line in text.splitlines():
    m = re.match(r'#define\s+(\w+)\s+\(?([\d\.f\-eE]+)\)?', line.strip())
    if m:
        val = m.group(2).rstrip('fF')
        try:
            defs[m.group(1)] = float(val)
        except:
            pass

print("Loaded calibration defines:", len(defs))
for k, v in list(defs.items())[:15]:
    print(f"  {k} = {v}")
