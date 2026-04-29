"""interactive_seg src package.

Top-level monkey-patch for torchio 0.19.6+ regression: ``ScalarImage.__init__``
in newer torchio rejects ``type`` kwarg (because ScalarImage type is always
``INTENSITY``), but its own ``__copy__`` still passes ``type`` to the new
instance. This breaks ``DataLoader(pin_memory=True)`` whenever a Subject is
yielded.

Patch drops the ``type`` kwarg before delegating, since it's a no-op for
ScalarImage.
"""

try:
    import torchio as _tio

    _orig_scalar_init = _tio.ScalarImage.__init__

    def _patched_scalar_init(self, *args, **kwargs):
        kwargs.pop("type", None)
        _orig_scalar_init(self, *args, **kwargs)

    _tio.ScalarImage.__init__ = _patched_scalar_init
except Exception:  # pragma: no cover
    pass
