"""Text-normalization ownership and cache identity (no model imports)."""
from importlib.metadata import PackageNotFoundError, version


def normalization_identity(engine):
    if engine != 'cosyvoice':
        return {'owner': 'legacy-app', 'revision': 1}
    packages = {}
    for name in ('ttsfrd', 'wetext'):
        try:
            packages[name] = version(name)
        except PackageNotFoundError:
            packages[name] = None
    return {'owner': 'cosyvoice-native', 'revision': 1, 'packages': packages}


def require_native_frontend(model):
    backend = getattr(getattr(model, 'frontend', None), 'text_frontend', '')
    if backend not in {'ttsfrd', 'wetext'}:
        raise RuntimeError(
            'CosyVoice requires its native ttsfrd or wetext text frontend. '
            'Install CosyVoice/requirements.txt and check frontend initialization logs; '
            'simplified normalization is disabled in OmniDub.'
        )
    return backend
