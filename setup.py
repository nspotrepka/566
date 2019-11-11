import gdown
import io
import os
import requests
from tqdm import tqdm
import zipfile

gaped_url = 'https://www.unige.ch/cisa/index.php/download_file/view/288/296/'
pmemo_url = 'https://drive.google.com/uc?id=1UzC3NCDj30j9Ba7i5lkMzWO5gFqSr0OJ'

directory = 'data/'
gaped_file = 'data/GAPED.zip'
pmemo_file = 'data/PMEmo2019.zip'

request = requests.get(gaped_url, stream=True)
with open(gaped_file, 'wb') as f:
    length = int(request.headers.get('content-length'))
    chunk_size = 512 * 1024
    t = tqdm(total=length, unit='B', unit_scale=True, desc='Downloading GAPED')
    for chunk in request.iter_content(chunk_size=chunk_size):
        if chunk:
            t.update(chunk_size)
            f.write(chunk)
            f.flush()

with zipfile.ZipFile(gaped_file) as zf:
    for member in tqdm(zf.infolist(), desc='Extracting GAPED'):
        try:
            zf.extract(member, 'data/')
        except zipfile.error as e:
            pass

print('Downloading PMEmo:')
gdown.download(pmemo_url, pmemo_file, quiet=False)

with zipfile.ZipFile(pmemo_file) as zf:
    for member in tqdm(zf.infolist(), desc='Extracting PMEmo'):
        try:
            if not member.filename.startswith('__MACOSX'):
                zf.extract(member, 'data/')
        except zipfile.error as e:
            pass

os.remove(gaped_file)
os.remove(pmemo_file)
