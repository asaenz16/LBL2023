"""Fetch only EMNIST Balanced members from the official NIST ZIP via HTTP ranges.

No mirror, credentials, full ZIP download, or dataset package is needed. ZIP CRCs,
IDX headers, counts and SHA256 hashes are checked/recorded. Cache stays in results/.
"""
import argparse
import gzip
import hashlib
import io
import json
import struct
import subprocess
import zipfile
from pathlib import Path
import numpy as np

URL = 'https://biometrics.nist.gov/cs_links/EMNIST/gzip.zip'


class RangeFile(io.RawIOBase):
    def __init__(self, url):
        self.url = url
        headers = subprocess.run(['curl', '-fsSI', '--max-time', '60', url],
                                 check=True, capture_output=True).stdout.decode()
        self.size = int(next(line.split(':', 1)[1] for line in headers.splitlines()
                             if line.lower().startswith('content-length:')))
        self.pos = 0

    def seekable(self): return True

    def seek(self, offset, whence=0):
        self.pos = offset if whence == 0 else self.pos+offset if whence == 1 else self.size+offset
        return self.pos

    def tell(self): return self.pos

    def read(self, size=-1):
        size = self.size-self.pos if size < 0 else min(size, self.size-self.pos)
        if size <= 0: return b''
        start = self.pos
        response = subprocess.run(['curl', '-fsSL', '--max-time', '120', '--max-filesize',
            '40000000', '--range', f'{start}-{start+size-1}', '--write-out', '%{http_code}',
            self.url], check=True, capture_output=True).stdout
        if response[-3:] != b'206':
            raise RuntimeError('Server ignored range; refusing an implicit full-archive download.')
        data = response[:-3]
        if len(data) != size: raise RuntimeError('Truncated HTTP range.')
        self.pos += size
        return data


def load_idx(path):
    data = gzip.decompress(Path(path).read_bytes())
    magic = struct.unpack('>I', data[:4])[0]
    if magic == 2051:
        n, rows, cols = struct.unpack('>III', data[4:16])
        if (rows, cols) != (28, 28) or len(data) != 16+n*rows*cols:
            raise ValueError('Unexpected EMNIST image dimensions or length.')
        # Official IDX orientation is transposed; return upright, contiguous images.
        return np.frombuffer(data, np.uint8, offset=16).reshape(n, rows, cols).transpose(0, 2, 1).copy()
    if magic == 2049:
        n = struct.unpack('>I', data[4:8])[0]
        if len(data) != 8+n: raise ValueError('Unexpected label length.')
        return np.frombuffer(data, np.uint8, offset=8).copy()
    raise ValueError('Unexpected IDX magic.')


def fetch(output):
    output.mkdir(parents=True, exist_ok=True)
    names = [f'emnist-balanced-{split}-{kind}-idx{dim}-ubyte.gz'
             for split in ('train', 'test') for kind, dim in (('images', 3), ('labels', 1))]
    names += ['emnist-balanced-mapping.txt']
    missing = [name for name in names if not (output/name).exists()]
    provenance = {'source': URL, 'dataset': 'EMNIST Balanced', 'orientation': 'transpose IDX image',
                  'citation': 'https://arxiv.org/abs/1702.05373', 'files': {}}
    if missing:
        with zipfile.ZipFile(RangeFile(URL)) as archive:
            for name in missing:
                members = [i for i in archive.infolist() if Path(i.filename).name == name]
                if len(members) != 1: raise RuntimeError(f'ZIP member ambiguous or absent: {name}')
                data = archive.read(members[0])  # ZipFile verifies the CRC.
                (output/name).write_bytes(data)
                print(f'Fetched {name}: {len(data):,} bytes; ZIP CRC verified', flush=True)
    for name in names:
        data = (output/name).read_bytes()
        provenance['files'][name] = {'bytes': len(data), 'sha256': hashlib.sha256(data).hexdigest()}
    for split, count in [('train', 112800), ('test', 18800)]:
        images = load_idx(output/f'emnist-balanced-{split}-images-idx3-ubyte.gz')
        labels = load_idx(output/f'emnist-balanced-{split}-labels-idx1-ubyte.gz')
        assert images.shape == (count, 28, 28) and labels.shape == (count,)
        assert np.array_equal(np.unique(labels), np.arange(47))
        assert np.all(np.bincount(labels) == count//47)
    (output/'provenance.json').write_text(json.dumps(provenance, indent=2)+'\n')
    return provenance


if __name__ == '__main__':
    p = argparse.ArgumentParser(); p.add_argument('--output', type=Path,
        default=Path(__file__).resolve().parents[1]/'results'/'emnist-data')
    print(json.dumps(fetch(p.parse_args().output), indent=2))
