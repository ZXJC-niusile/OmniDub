"""Native TN boundary tests; synthesis and tokenizer weights are not required."""
import ast
from functools import partial
import importlib.util
import json
import logging
from pathlib import Path
import re
import sys
import tempfile
import types
from typing import Generator
import unittest
from unittest import mock

# Reuse the existing optional GPU dependency stubs.
import test_omnidub_regressions  # noqa: F401
from youdub import step040_tts as tts, step043_tts_cosyvoice as cv
from youdub import text_frontend as policy

ROOT = Path(__file__).resolve().parents[1]


class NativeFrontendTests(unittest.TestCase):
    def test_cosyvoice_receives_original_notation(self):
        texts = ['误差0.05，特工007', '2024-05-01晚上20:30',
                 'GPT-4、A100、v1.02、P2P', '比分2-1，4-4-2阵型',
                 'The price is $12.50.', '说明（版本1.02）']
        with mock.patch.object(tts, 'normalizer', side_effect=AssertionError('duplicate TN')):
            for text in texts:
                self.assertEqual(tts.preprocess_text(text, tts_engine='cosyvoice'), text)
        for text in ['', '...', '（音乐）']:
            self.assertEqual(tts.preprocess_text(text, tts_engine='cosyvoice'), '')

    def test_missing_native_backend_is_an_error(self):
        for backend in ('', None, 'unknown'):
            model = types.SimpleNamespace(frontend=types.SimpleNamespace(text_frontend=backend))
            with self.assertRaisesRegex(RuntimeError, 'CosyVoice/requirements.txt'):
                policy.require_native_frontend(model)
        for backend in ('wetext', 'ttsfrd'):
            model = types.SimpleNamespace(frontend=types.SimpleNamespace(text_frontend=backend))
            self.assertEqual(policy.require_native_frontend(model), backend)

    def test_backend_version_changes_both_cache_layers(self):
        from youdub import pipeline
        with tempfile.TemporaryDirectory() as folder:
            wavs = Path(folder) / 'wavs'
            with mock.patch.object(policy, 'version', return_value='old'):
                before = pipeline._tts_spec(folder, pipeline.PipelineConfig(tts_engine='cosyvoice'))[1]
                tts._prepare_segment_wav_cache(str(wavs), {'text_normalization': policy.normalization_identity('cosyvoice')})
            clip = wavs / '0000.wav'
            clip.write_bytes(b'old speech')
            with mock.patch.object(policy, 'version', return_value='new'):
                after = pipeline._tts_spec(folder, pipeline.PipelineConfig(tts_engine='cosyvoice'))[1]
                tts._prepare_segment_wav_cache(str(wavs), {'text_normalization': policy.normalization_identity('cosyvoice')})
            self.assertNotEqual(before, after)
            self.assertFalse(clip.exists())

    def test_failed_frontend_is_not_cached_as_a_loaded_model(self):
        module = types.ModuleType('cosyvoice.cli.cosyvoice')
        module.CosyVoice = mock.Mock(return_value=types.SimpleNamespace(
            frontend=types.SimpleNamespace(text_frontend='')))
        with mock.patch.dict(sys.modules, {'cosyvoice.cli.cosyvoice': module}), \
             mock.patch.dict(cv._cosyvoice_models, clear=True), \
             mock.patch.object(cv, '_bind_cosyvoice_device') as bind:
            with self.assertRaisesRegex(RuntimeError, 'native ttsfrd or wetext'):
                cv.load_cosyvoice_model('cpu')
            self.assertEqual(cv._cosyvoice_models, {})
            bind.assert_not_called()

    def test_all_inference_routes_enable_native_frontend(self):
        tensor = mock.Mock()
        tensor.dim.return_value = 2
        for prompt, fail_zero in [('', False), ('参考0.05', False), ('参考0.05', True)]:
            model = mock.Mock(sample_rate=24000)
            model.inference_zero_shot.return_value = iter([{'tts_speech': tensor}])
            model.inference_cross_lingual.return_value = iter([{'tts_speech': tensor}])
            if fail_zero:
                model.inference_zero_shot.side_effect = RuntimeError('synthesis failed')
            with tempfile.TemporaryDirectory() as folder, \
                 mock.patch.object(cv, 'load_cosyvoice_model', return_value=model), \
                 mock.patch.object(cv, '_ensure_under_30s', return_value='ref.wav'), \
                 mock.patch.object(cv.torch, 'cat', return_value=tensor), \
                 mock.patch.object(cv.torchaudio, 'save'):
                cv.generate_cosyvoice_audio('误差0.05', 'ref.wav', str(Path(folder)/'out.wav'), prompt_text=prompt, device='cpu')
            if prompt:
                model.inference_zero_shot.assert_called_once_with('误差0.05', prompt, 'ref.wav', text_frontend=True)
            if not prompt or fail_zero:
                model.inference_cross_lingual.assert_called_once_with('误差0.05', 'ref.wav', text_frontend=True)
            else:
                model.inference_cross_lingual.assert_not_called()

    def test_real_wetext_through_bundled_native_method(self):
        try:
            from wetext import Normalizer
        except ImportError:
            self.skipTest('Install wetext==0.2.0 for the real TN integration test')
        # Execute the bundled text_normalize method unchanged, without importing
        # ONNX/Whisper or loading speech weights. Its text utilities are real.
        spec = importlib.util.spec_from_file_location('native_utils', ROOT/'CosyVoice/cosyvoice/utils/frontend_utils.py')
        utils = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(utils)
        tree = ast.parse((ROOT/'CosyVoice/cosyvoice/cli/frontend.py').read_text(encoding='utf-8'))
        cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == 'CosyVoiceFrontEnd')
        method = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == 'text_normalize')
        namespace = dict(vars(utils), Generator=Generator, logging=logging, json=json, partial=partial, re=re)
        exec(compile(ast.Module(body=[method], type_ignores=[]), str(ROOT/'CosyVoice/cosyvoice/cli/frontend.py'), 'exec'), namespace)
        zh = mock.Mock(wraps=Normalizer(lang='zh', operator='tn', remove_erhua=False))
        en = mock.Mock(wraps=Normalizer(lang='en', operator='tn'))
        frontend = types.SimpleNamespace(text_frontend='wetext', zh_tn_model=zh, en_tn_model=en,
                                        tokenizer=types.SimpleNamespace(encode=lambda text, **kw: list(text)),
                                        allowed_special='all', inflect_parser=mock.Mock())
        cases = [('误差0.05', '误差零点零五。', zh),
                 ('特工007', '特工零零七。', zh),
                 ('2024年5月1日', '二零二四年五月一日。', zh),
                 ('The price is $12.50.', 'The price is twelve point five dollars.', en)]
        for raw, expected, normalizer in cases:
            normalizer.reset_mock()
            prepared = tts.preprocess_text(raw, tts_engine='cosyvoice')
            result = namespace['text_normalize'](frontend, prepared)
            self.assertEqual(''.join(result), expected)
            normalizer.normalize.assert_called_once_with(raw)


if __name__ == '__main__':
    unittest.main()
