"""Check pinned software while accepting compatible CUDA hardware."""
import json

from training import environment


def test_cuda_hardware_is_reported_without_requiring_a_specific_model(monkeypatch):
    manifest = json.loads(environment.DEFAULT_MANIFEST.read_text())
    monkeypatch.setattr(environment, 'collect_runtime', lambda: manifest['runtime'])
    monkeypatch.setattr(environment, 'collect_platform', lambda: manifest['platform'])
    monkeypatch.setattr(environment.torch.cuda, 'is_available', lambda: True)
    monkeypatch.setattr(environment.torch.cuda, 'get_device_name', lambda _: 'NVIDIA RTX A6000')
    monkeypatch.setattr(environment, 'collect_nvidia_driver', lambda: '590.00')
    result = environment.validate_environment(require_cuda=True)
    assert result['compatible']
    assert result['hardware']['gpu'] == 'NVIDIA RTX A6000'
    monkeypatch.setattr(environment.torch.cuda, 'is_available', lambda: False)
    assert environment.validate_environment()['compatible']
    assert not environment.validate_environment(require_cuda=True)['compatible']
    changed = dict(manifest['runtime'], torch='different')
    monkeypatch.setattr(environment, 'collect_runtime', lambda: changed)
    assert 'torch' in environment.validate_environment()['mismatches']
