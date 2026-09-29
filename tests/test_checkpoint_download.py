import hashlib
import importlib.util
import io
from pathlib import Path
import tarfile

import pytest

spec = importlib.util.spec_from_file_location('download_checkpoints', Path(__file__).parents[1] / 'scripts/download_checkpoints.py')
download = importlib.util.module_from_spec(spec)
spec.loader.exec_module(download)


def archive(tmp_path, name='gridworld/model.pt', data=b'model'):
    path = tmp_path / 'bundle.tar'
    with tarfile.open(path, 'w') as bundle:
        entry = tarfile.TarInfo(name)
        entry.size = len(data)
        bundle.addfile(entry, io.BytesIO(data))
    row = dict(path=name, sha256=hashlib.sha256(data).hexdigest(), bytes=len(data))
    return path, row


def test_installs_verified_files_reuses_identical_and_refuses_conflict(tmp_path):
    bundle, row = archive(tmp_path)
    output = tmp_path / 'out'; output.mkdir()
    download.extract_verified(bundle, output, [row])
    download.extract_verified(bundle, output, [row])
    (output / row['path']).write_bytes(b'other')
    with pytest.raises(ValueError, match='existing file differs'):
        download.extract_verified(bundle, output, [row])
    assert (output / row['path']).read_bytes() == b'other'


def test_corruption_and_path_escape_are_rejected_before_install(tmp_path):
    bundle, row = archive(tmp_path)
    output = tmp_path / 'out'; output.mkdir()
    with pytest.raises(ValueError, match='checksum'):
        download.extract_verified(bundle, output, [{**row, 'sha256': '0'*64}])
    assert not (output / row['path']).exists()
    bundle, row = archive(tmp_path, '../outside')
    with pytest.raises(ValueError, match='unsafe'):
        download.extract_verified(bundle, output, [row])
