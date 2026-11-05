import hashlib,json,time,subprocess,sys
from pathlib import Path
root=Path(__file__).resolve().parents[1]
expected={'seed42.pt': 'a2f1ac437eed16b5fc1ec9fdf5c4cee12d469918847680b156f94a4b9f4278ba', 'seed43.pt': 'c151242093a3e3c41b9ae881f718852804241584564127b0a4ed5ebb4526d695', 'seed44.pt': '2f5b213a10eccc7e4a636dcfc67270778636a811d12b00d5b9394cee0711de00', 'runtime_sample.json': '9e87b20eec17b1889fb56e5320609e3c46c89252ae47e90904e3c11c8204ae99'}
for attempt in range(360):
    ready=all((root/'inputs'/name).is_file() and hashlib.sha256((root/'inputs'/name).read_bytes()).hexdigest()==digest for name,digest in expected.items())
    if ready:break
    time.sleep(10)
else:raise RuntimeError('Input transfer hashes not ready after one hour')
print('INPUT_HASHES_PASS',flush=True)
for script in ('verify_erasing_native.py','check_erasing_runtime.py'):
    subprocess.run([sys.executable,'-u',str(root/'scripts'/script)],cwd=root,check=True)
