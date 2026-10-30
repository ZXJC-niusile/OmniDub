# -*- coding: utf-8 -*-
"""Regression unit test suite for OmniDub pipeline, scheduling, audio/text/video helpers, and UI gates."""

from concurrent.futures import ThreadPoolExecutor
from contextlib import nullcontext
import json
import os
import py_compile
import sys
import tempfile
import time
import types
import unittest
from unittest import mock
import wave

import numpy as np

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)


def _install_stub_modules_if_missing() -> None:
    """Install lightweight stubs only when optional GPU/ML packages are absent in bare Python."""

    def _ensure(name: str, attrs: dict | None = None):
        try:
            return __import__(name, fromlist=['*'])
        except ImportError:
            mod = types.ModuleType(name)
            for k, v in (attrs or {}).items():
                setattr(mod, k, v)
            sys.modules[name] = mod
            return mod

    try:
        import loguru  # noqa: F401
    except ImportError:
        class _DummyLogger:
            def debug(self, *a, **k): pass
            def info(self, *a, **k): pass
            def warning(self, *a, **k): pass
            def error(self, *a, **k): pass
            def exception(self, *a, **k): pass

        _ensure('loguru', {'logger': _DummyLogger()})

    try:
        import scipy.io.wavfile  # noqa: F401
    except ImportError:
        def _write_wav_stdlib(filename, rate, data):
            arr = np.asarray(data, dtype=np.int16)
            channels = 1 if arr.ndim <= 1 else arr.shape[1]
            with wave.open(str(filename), 'wb') as wf:
                wf.setnchannels(channels)
                wf.setsampwidth(2)
                wf.setframerate(int(rate))
                wf.writeframes(arr.tobytes())

        scipy_mod = _ensure('scipy')
        io_mod = _ensure('scipy.io')
        wav_mod = _ensure('scipy.io.wavfile', {'write': _write_wav_stdlib})
        scipy_mod.io = io_mod
        io_mod.wavfile = wav_mod

    try:
        import torch  # noqa: F401
    except ImportError:
        class _FakeTorchDevice:
            def __init__(self, spec):
                s = str(spec)
                if ':' in s:
                    self.type, idx_s = s.split(':', 1)
                    self.index = int(idx_s)
                else:
                    self.type = s
                    self.index = None

            def __str__(self):
                return f'{self.type}:{self.index}' if self.index is not None else self.type

            def __eq__(self, other):
                return str(self) == str(other)

        class _Cuda:
            @staticmethod
            def is_available():
                return False

            @staticmethod
            def device_count():
                return 0

            @staticmethod
            def empty_cache():
                pass

            @staticmethod
            def device(idx):
                return nullcontext()

            @staticmethod
            def Stream(dev=None):
                return object()

            @staticmethod
            def stream(s):
                return nullcontext()

        class _Serialization:
            @staticmethod
            def add_safe_globals(g):
                pass

        _ensure(
            'torch',
            {
                'cuda': _Cuda(),
                'device': _FakeTorchDevice,
                'load': lambda *a, **k: {},
                'serialization': _Serialization(),
                'zeros': lambda *a, **k: None,
                'cat': lambda *a, **k: None,
                'nn': types.SimpleNamespace(Module=object, functional=types.SimpleNamespace()),
            },
        )

    _ensure('torchaudio', {'load': lambda p: (None, 24000), 'save': lambda *a, **k: None})
    _ensure('dotenv', {'load_dotenv': lambda *a, **k: None})
    _ensure('yt_dlp', {'YoutubeDL': object})
    demucs_mod = _ensure('demucs')
    demucs_api = _ensure('demucs.api', {'Separator': object})
    demucs_mod.api = demucs_api
    _ensure(
        'whisperx',
        {
            'load_model': lambda *a, **k: None,
            'load_align_model': lambda *a, **k: (None, None),
            'align': lambda *a, **k: {},
            'assign_word_speakers': lambda *a, **k: {},
        },
    )
    _ensure('librosa', {'load': lambda *a, **k: (np.zeros(2400, dtype=np.float32), 24000), 'get_duration': lambda *a, **k: 1.0})
    _ensure('openai', {'OpenAI': object})
    tts_mod = _ensure('TTS')
    tts_api = _ensure('TTS.api', {'TTS': object})
    tts_mod.api = tts_api
    as_mod = _ensure('audiostretchy')
    as_stretch = _ensure('audiostretchy.stretch', {'stretch_audio': lambda *a, **k: None})
    as_mod.stretch = as_stretch
    _ensure(
        'gradio',
        {
            'Interface': lambda **k: k,
            'TabbedInterface': lambda **k: k,
            'Textbox': lambda *a, **k: k,
            'Radio': lambda *a, **k: k,
            'Slider': lambda *a, **k: k,
            'Dropdown': lambda *a, **k: k,
            'Checkbox': lambda *a, **k: k,
        },
    )
    try:
        import PIL.Image  # noqa: F401
    except ImportError:
        pil_mod = _ensure('PIL')
        img_mod = _ensure('PIL.Image')
        pil_mod.Image = img_mod


_install_stub_modules_if_missing()


def _read_wav_int16(path: str) -> tuple[int, np.ndarray]:
    with wave.open(path, 'rb') as wf:
        sr = wf.getframerate()
        raw = wf.readframes(wf.getnframes())
    return sr, np.frombuffer(raw, dtype=np.int16)


class OmniDubRegressionTests(unittest.TestCase):

    def test_01_compile_all_python_files_and_shell_script_hygiene(self):
        compiled = 0
        for root, dirs, files in os.walk(REPO_ROOT):
            dirs[:] = [d for d in dirs if d not in {'.git', '__pycache__', '.venv', 'venv'}]
            for fn in files:
                if fn.endswith('.py'):
                    py_compile.compile(os.path.join(root, fn), doraise=True)
                    compiled += 1
        self.assertGreaterEqual(compiled, 30)

        install_sh = os.path.join(REPO_ROOT, 'scripts', 'install_all.sh')
        with open(install_sh, 'rb') as f:
            sh_bytes = f.read()
        self.assertNotIn(b'\r\n', sh_bytes, 'scripts/install_all.sh must use LF line endings')
        sh_text = sh_bytes.decode('utf-8')
        self.assertIn('REPO_ROOT=', sh_text)
        self.assertIn('INSTALL_XTTS="${INSTALL_XTTS:-0}"', sh_text)

        gitattributes = os.path.join(REPO_ROOT, '.gitattributes')
        self.assertTrue(os.path.isfile(gitattributes))
        with open(gitattributes, 'r', encoding='utf-8') as f:
            self.assertIn('*.sh text eol=lf', f.read())

    def test_02_dag_cache_cascade_invalidation_across_restart(self):
        from youdub import pipeline, utils

        with tempfile.TemporaryDirectory() as tmp:
            # Seed initial upstream video and SPEAKER reference wav
            with open(os.path.join(tmp, 'download.mp4'), 'wb') as f:
                f.write(b'video_v1' * 200)
            with open(os.path.join(tmp, 'download.info.json'), 'w', encoding='utf-8') as f:
                json.dump({'title': 'Demo', 'uploader': 'Tester'}, f)
            os.makedirs(os.path.join(tmp, 'SPEAKER'), exist_ok=True)
            utils.save_wav(np.ones(2400, dtype=np.float32) * 0.1, os.path.join(tmp, 'SPEAKER', 'SPEAKER_00.wav'))

            calls = []

            def _fake_extract(folder):
                calls.append('s1_extract')
                utils.save_wav(np.ones(2400, dtype=np.float32) * 0.1, os.path.join(folder, 'audio.wav'))

            def _fake_separate(folder, **kw):
                calls.append('s1_separate')
                utils.save_wav(np.ones(2400, dtype=np.float32) * 0.1, os.path.join(folder, 'audio_vocals.wav'))
                utils.save_wav(np.ones(2400, dtype=np.float32) * 0.05, os.path.join(folder, 'audio_instruments.wav'))

            def _fake_transcribe(folder, **kw):
                calls.append('s2_transcribe')
                os.makedirs(os.path.join(folder, 'SPEAKER'), exist_ok=True)
                utils.save_wav(
                    np.ones(2400, dtype=np.float32) * 0.1,
                    os.path.join(folder, 'SPEAKER', 'SPEAKER_00.wav'),
                )
                with open(os.path.join(folder, 'transcript.json'), 'w', encoding='utf-8') as f:
                    json.dump([{'start': 0.0, 'end': 1.0, 'text': 'hello', 'speaker': 'SPEAKER_00'}], f)

            def _fake_translate(folder, **kw):
                calls.append('s3_translate')
                with open(os.path.join(folder, 'summary.json'), 'w', encoding='utf-8') as f:
                    json.dump({'title': '标题', 'summary': '摘要', 'language': '简体中文'}, f)
                with open(os.path.join(folder, 'translation.json'), 'w', encoding='utf-8') as f:
                    json.dump([{'start': 0.0, 'end': 1.0, 'text': 'hello', 'translation': '你好', 'speaker': 'SPEAKER_00'}], f)

            def _fake_tts(folder, **kw):
                calls.append('s4_tts')
                utils.save_wav(np.ones(2400, dtype=np.float32) * 0.1, os.path.join(folder, 'audio_combined.wav'))
                with open(os.path.join(folder, 'translation_tts.json'), 'w', encoding='utf-8') as f:
                    json.dump([{'start': 0.0, 'end': 1.0, 'text': 'hello', 'translation': '你好', 'speaker': 'SPEAKER_00'}], f)

            def _fake_synth(folder, **kw):
                calls.append('s5_synth')
                with open(os.path.join(folder, 'video.mp4'), 'wb') as f:
                    f.write(b'final_mp4_bytes')

            def _fake_info(folder):
                calls.append('s6_info')
                with open(os.path.join(folder, 'video.txt'), 'w', encoding='utf-8') as f:
                    f.write('Demo\n\n摘要')

            cfg = pipeline.PipelineConfig(root_folder=tmp)
            with mock.patch.object(pipeline, 'extract_audio_from_video', side_effect=_fake_extract), \
                 mock.patch.object(pipeline, 'separate_audio', side_effect=_fake_separate), \
                 mock.patch.object(pipeline, 'transcribe_audio', side_effect=_fake_transcribe), \
                 mock.patch.object(pipeline, 'translate', side_effect=_fake_translate), \
                 mock.patch.object(pipeline, 'generate_wavs', side_effect=_fake_tts), \
                 mock.patch.object(pipeline, 'synthesize_video', side_effect=_fake_synth), \
                 mock.patch.object(pipeline, 'generate_info', side_effect=_fake_info):

                # Run 1: all 6 steps execute
                pipeline._run_pipeline(tmp, cfg)
                self.assertEqual(
                    calls,
                    ['s1_extract', 's1_separate', 's2_transcribe', 's3_translate', 's4_tts', 's5_synth', 's6_info'],
                )

                # Run 2 (nothing changed): all 6 steps skip
                calls.clear()
                pipeline._run_pipeline(tmp, cfg)
                self.assertEqual(calls, [])

                # Simulate editing/updating Step 2 transcript.json after restart:
                # Steps 1 & 2 remain valid, while Steps 3 -> 4 -> 5 -> 6 MUST cascade!
                with open(os.path.join(tmp, 'transcript.json'), 'w', encoding='utf-8') as f:
                    json.dump([{'start': 0.0, 'end': 1.5, 'text': 'hello world', 'speaker': 'SPEAKER_00'}], f)

                def _fake_translate_v2(folder, **kw):
                    calls.append('s3_translate_v2')
                    with open(os.path.join(folder, 'summary.json'), 'w', encoding='utf-8') as f:
                        json.dump({'title': '新标题', 'summary': '新摘要', 'language': '简体中文'}, f)
                    with open(os.path.join(folder, 'translation.json'), 'w', encoding='utf-8') as f:
                        json.dump(
                            [{'start': 0.0, 'end': 1.5, 'text': 'hello world', 'translation': '你好世界', 'speaker': 'SPEAKER_00'}],
                            f,
                        )

                def _fake_tts_v2(folder, **kw):
                    calls.append('s4_tts_v2')
                    utils.save_wav(np.ones(3600, dtype=np.float32) * 0.2, os.path.join(folder, 'audio_combined.wav'))
                    with open(os.path.join(folder, 'translation_tts.json'), 'w', encoding='utf-8') as f:
                        json.dump(
                            [{'start': 0.0, 'end': 1.4, 'text': 'hello world', 'translation': '你好世界', 'speaker': 'SPEAKER_00'}],
                            f,
                        )

                with mock.patch.object(pipeline, 'translate', side_effect=_fake_translate_v2), \
                     mock.patch.object(pipeline, 'generate_wavs', side_effect=_fake_tts_v2):
                    pipeline._run_pipeline(tmp, cfg)

                self.assertEqual(calls, ['s3_translate_v2', 's4_tts_v2', 's5_synth', 's6_info'])

    def test_03_cuda_visible_devices_mapping_and_ordinal_clamping(self):
        from youdub import utils

        smi_rows = [
            ('0', 'GPU-0000-aaaa', 14000),
            ('1', 'GPU-1111-bbbb', 12000),
            ('2', 'GPU-2222-cccc', 300),
            ('3', 'GPU-3333-dddd', 9000),
        ]

        with mock.patch.object(utils.torch.cuda, 'is_available', return_value=True), \
             mock.patch.object(utils.torch.cuda, 'device_count', return_value=2), \
             mock.patch.object(utils, '_query_nvidia_smi_rows', return_value=smi_rows), \
             mock.patch.dict(os.environ, {'CUDA_VISIBLE_DEVICES': '2,3'}, clear=False):
            utils._RESERVED.clear()
            # Physical GPU 2 (300 MB) is logical cuda:0; physical GPU 3 (9000 MB) is logical cuda:1!
            self.assertEqual(utils.pick_least_used_cuda_device(), 0)
            with utils.reserved_cuda_devices(2) as devs:
                self.assertEqual(devs, ['cuda:0', 'cuda:1'])
            self.assertEqual(utils._RESERVED, {})

        # UUID prefix mapping in reversed order: GPU-3333 -> logical cuda:0 (9000MB), GPU-2222 -> logical cuda:1 (300MB)
        with mock.patch.object(utils.torch.cuda, 'is_available', return_value=True), \
             mock.patch.object(utils.torch.cuda, 'device_count', return_value=2), \
             mock.patch.object(utils, '_query_nvidia_smi_rows', return_value=smi_rows), \
             mock.patch.dict(os.environ, {'CUDA_VISIBLE_DEVICES': 'GPU-3333,GPU-2222'}, clear=False):
            self.assertEqual(utils.pick_least_used_cuda_device(), 1)

        # Invalid device token halts enumeration after first valid entry (CUDA runtime semantics)
        with mock.patch.object(utils.torch.cuda, 'is_available', return_value=True), \
             mock.patch.object(utils.torch.cuda, 'device_count', return_value=1), \
             mock.patch.object(utils, '_query_nvidia_smi_rows', return_value=smi_rows), \
             mock.patch.dict(os.environ, {'CUDA_VISIBLE_DEVICES': '3,99,2'}, clear=False):
            with utils.reserved_cuda_devices(2) as devs:
                self.assertEqual(devs, ['cuda:0', 'cuda:0'])

        # Masked empty CUDA_VISIBLE_DEVICES falls back to CPU safely
        with mock.patch.object(utils.torch.cuda, 'is_available', return_value=True), \
             mock.patch.object(utils.torch.cuda, 'device_count', return_value=0), \
             mock.patch.object(utils, '_query_nvidia_smi_rows', return_value=smi_rows), \
             mock.patch.dict(os.environ, {'CUDA_VISIBLE_DEVICES': ''}, clear=False):
            with utils.reserved_cuda_devices(2) as devs:
                self.assertEqual(devs, ['cpu'])

        # resolve_torch_device invalidates stale ordinal when CUDA_VISIBLE_DEVICES shrinks
        utils._AUTO_CUDA_DEVICE = 'cuda:3'
        utils._AUTO_CUDA_VISIBLE_ENV = '0,1,2,3'
        with mock.patch.object(utils.torch.cuda, 'is_available', return_value=True), \
             mock.patch.object(utils.torch.cuda, 'device_count', return_value=1), \
             mock.patch.object(utils, '_query_nvidia_smi_rows', return_value=[('2', 'GPU-2', 100)]), \
             mock.patch.dict(os.environ, {'CUDA_VISIBLE_DEVICES': '2'}, clear=False):
            self.assertEqual(utils.resolve_torch_device('auto'), 'cuda:0')

    def test_04_save_wav_clipping_and_empty_array_safety(self):
        from youdub import utils

        with tempfile.TemporaryDirectory() as tmp:
            clip_path = os.path.join(tmp, 'clipped.wav')
            samples = np.array([1.5, -1.5, 0.0, np.nan, np.inf, -np.inf], dtype=np.float32)
            utils.save_wav(samples, clip_path, sample_rate=24000)
            sr, data = _read_wav_int16(clip_path)
            self.assertEqual(sr, 24000)
            self.assertEqual(data.tolist(), [32767, -32767, 0, 0, 32767, -32767])

            empty_a = os.path.join(tmp, 'empty_a.wav')
            empty_b = os.path.join(tmp, 'empty_b.wav')
            utils.save_wav(np.zeros((0,), dtype=np.float32), empty_a, sample_rate=24000)
            utils.save_wav_norm(np.zeros((0,), dtype=np.float32), empty_b, sample_rate=24000)
            _, data_a = _read_wav_int16(empty_a)
            _, data_b = _read_wav_int16(empty_b)
            self.assertEqual(len(data_a), 0)
            self.assertEqual(len(data_b), 0)

    def test_05_tts_subtitle_end_excludes_post_pad_and_handles_punctuation_only(self):
        from youdub import step040_tts as st_tts
        from youdub import utils

        with tempfile.TemporaryDirectory() as tmp:
            os.makedirs(os.path.join(tmp, 'SPEAKER'), exist_ok=True)
            utils.save_wav(np.ones(4800, dtype=np.float32) * 0.1, os.path.join(tmp, 'SPEAKER', 'SPEAKER_00.wav'))
            utils.save_wav(np.ones(24000 * 15, dtype=np.float32) * 0.2, os.path.join(tmp, 'audio_vocals.wav'))
            utils.save_wav(np.ones(24000 * 15, dtype=np.float32) * 0.05, os.path.join(tmp, 'audio_instruments.wav'))

            # Line 0 ends at 1.0s, followed by a 10s BGM gap before Line 1 at 11.0s;
            # Line 2 is punctuation-only "...", Line 3 is stage-direction "（掌声）" with speaker=None.
            orig_data = [
                {'start': 0.0, 'end': 1.0, 'text': 'Hello', 'translation': '你好', 'speaker': 'SPEAKER_00'},
                {'start': 11.0, 'end': 12.0, 'text': 'World', 'translation': '世界', 'speaker': 'SPEAKER_00'},
                {'start': 12.5, 'end': 13.0, 'text': '...', 'translation': '...', 'speaker': 'SPEAKER_00'},
                {'start': 13.2, 'end': 14.0, 'text': '(Applause)', 'translation': '（掌声）', 'speaker': None},
            ]
            tr_path = os.path.join(tmp, 'translation.json')
            with open(tr_path, 'w', encoding='utf-8') as f:
                json.dump(orig_data, f)

            synthesized_texts = []

            def _fake_tts_func(text, out_path, spk_wav, spk_id):
                synthesized_texts.append((text, spk_id))
                utils.save_wav(np.ones(int(0.8 * 24000), dtype=np.float32) * 0.2, out_path, sample_rate=24000)

            def _fake_get_duration(path=None, sr=24000, **kw):
                _, pcm = _read_wav_int16(path)
                return len(pcm) / float(sr)

            def _fake_librosa_load(path, sr=24000, **kw):
                _, pcm = _read_wav_int16(path)
                return pcm.astype(np.float32) / 32767.0, sr

            with mock.patch.object(st_tts, '_select_tts_func', return_value=_fake_tts_func), \
                 mock.patch.object(st_tts.librosa, 'get_duration', side_effect=_fake_get_duration), \
                 mock.patch.object(st_tts.librosa, 'load', side_effect=_fake_librosa_load):
                st_tts.generate_wavs(tmp, tts_engine='cosyvoice', target_language='简体中文')

            # Only speakable lines invoke TTS; "..." and "（掌声）" become clean silence
            self.assertEqual(synthesized_texts, [('你好', 'SPEAKER_00'), ('世界', 'SPEAKER_00')])

            # Step 3 translation.json remains untouched
            with open(tr_path, 'r', encoding='utf-8') as f:
                self.assertEqual(json.load(f), orig_data)

            with open(os.path.join(tmp, 'translation_tts.json'), 'r', encoding='utf-8') as f:
                aligned = json.load(f)

            # Line 0 speech is 0.8s: line['end'] MUST be ~0.8s (NOT 11.0s across the 10s gap!)
            self.assertAlmostEqual(aligned[0]['start'], 0.0, places=2)
            self.assertLessEqual(aligned[0]['end'], 1.0)
            # Line 1 still starts right on schedule at ~11.0s!
            self.assertAlmostEqual(aligned[1]['start'], 11.0, places=2)
            self.assertAlmostEqual(aligned[1]['end'], 11.8, places=2)
            self.assertEqual(aligned[3]['speaker'], 'SPEAKER_00')

            # adjust_audio_length path construction check for dotted filenames
            dotted_wav = os.path.join(tmp, 'seg.v1.wav')
            utils.save_wav(np.ones(24000, dtype=np.float32) * 0.1, dotted_wav, sample_rate=24000)
            stretched_targets = []

            def _fake_stretch(src, dst, ratio=1.0, sample_rate=24000):
                stretched_targets.append(dst)
                utils.save_wav(np.ones(int(24000 * ratio), dtype=np.float32) * 0.1, dst, sample_rate=sample_rate)

            with mock.patch.object(st_tts, 'stretch_audio', side_effect=_fake_stretch), \
                 mock.patch.object(st_tts.librosa, 'get_duration', return_value=1.0), \
                 mock.patch.object(st_tts.librosa, 'load', side_effect=_fake_librosa_load):
                # speed_factor == 1.0 bypasses stretch_audio completely
                st_tts.adjust_audio_length(dotted_wav, 1.2)
                self.assertEqual(stretched_targets, [])
                # speed_factor < 1.0 uses os.path.splitext -> seg.v1_adjusted.wav
                st_tts.adjust_audio_length(dotted_wav, 0.85)
                self.assertEqual(stretched_targets, [os.path.join(tmp, 'seg.v1_adjusted.wav')])

    def test_06_chinese_text_normalization_dates_times_ranges_scores_and_zeros(self):
        from youdub import step040_tts as st_tts

        self.assertEqual(
            st_tts.preprocess_text('2024-05-01晚上20:30直播', '简体中文'),
            '二零二四年五月一日晚上二十点三十分直播',
        )
        self.assertEqual(
            st_tts.preprocess_text('北京时间08:00开球', '简体中文'),
            '北京时间八点整开球',
        )
        self.assertEqual(
            st_tts.preprocess_text('只需3-5分钟，适合18-25岁人群', '简体中文'),
            '只需三到五分钟，适合十八到二十五岁人群',
        )
        self.assertEqual(
            st_tts.preprocess_text('比赛第10-15分钟', '简体中文'),
            '比赛第十到十五分钟',
        )
        self.assertEqual(
            st_tts.preprocess_text('他是00后球员', '简体中文'),
            '他是零零后球员',
        )
        tactical = st_tts.preprocess_text('他们踢4-4-2阵型，最终以2-1获胜', '简体中文')
        self.assertIn('四四二', tactical)
        self.assertIn('二比一', tactical)
        self.assertEqual(st_tts.preprocess_text('...', '简体中文'), '')
        self.assertEqual(st_tts.preprocess_text('（音乐）', '简体中文'), '')

    def test_07_subtitle_wrapping_zero_char_loss_and_portrait_video_layout(self):
        from youdub import step050_synthesize_video as syn

        # Verify 0 dropped characters across every CJK length from 1 to 100 (incl. 45, 49, 53, 57, 61)
        alphabet = '零一二三四五六七八九十甲乙丙丁戊己庚辛壬癸'
        for n in range(1, 101):
            src = ''.join(alphabet[i % len(alphabet)] for i in range(n))
            wrapped = syn.wrap_text(src, max_line_char=30)
            rejoined = ''.join(wrapped.split('\n'))
            self.assertEqual(rejoined, src, f'Characters lost at length={n}')
            for line in wrapped.split('\n'):
                self.assertLessEqual(len(line), 30)

        # Verify space-delimited English wraps on word boundaries without losing words
        en = 'Artificial intelligence dubbing preserves natural speaker timbre across languages.'
        wrapped_en = syn.wrap_text(en, max_line_char=30)
        self.assertEqual(' '.join(wrapped_en.split('\n')), en)
        for line in wrapped_en.split('\n'):
            self.assertLessEqual(len(line), 30)

        # Portrait 9:16 vs Landscape 16:9 subtitle layout + ffmpeg command verification
        p_char, p_orig, p_font = syn._subtitle_layout_for_aspect(9 / 16)
        l_char, l_orig, l_font = syn._subtitle_layout_for_aspect(16 / 9)
        self.assertLess(p_char, l_char)
        self.assertLess(p_orig, l_orig)
        self.assertLess(p_font, l_font)

        with tempfile.TemporaryDirectory() as tmp:
            with open(os.path.join(tmp, 'download.mp4'), 'wb') as f:
                f.write(b'mp4')
            with open(os.path.join(tmp, 'audio_combined.wav'), 'wb') as f:
                f.write(b'wav')
            with open(os.path.join(tmp, 'translation_tts.json'), 'w', encoding='utf-8') as f:
                json.dump([{'start': 0.0, 'end': 2.0, 'text': 'Hello world', 'translation': '你好世界'}], f)

            captured_cmd = []

            def _fake_run(cmd, **kw):
                captured_cmd.append(cmd)
                return types.SimpleNamespace(returncode=0, stdout='')

            with mock.patch.object(syn, 'get_aspect_ratio', return_value=9 / 16), \
                 mock.patch.object(syn.subprocess, 'run', side_effect=_fake_run):
                syn.synthesize_video(tmp, subtitles=True, speed_up=1.05, fps=30, resolution='1080p')

            self.assertEqual(len(captured_cmd), 1)
            cmd = captured_cmd[0]
            self.assertIn('-pix_fmt', cmd)
            self.assertEqual(cmd[cmd.index('-pix_fmt') + 1], 'yuv420p')
            fc = cmd[cmd.index('-filter_complex') + 1]
            self.assertLess(fc.index('scale=1080:1920'), fc.index("subtitles='"))
            self.assertIn('WrapStyle=0', fc)

    def test_08_downloader_resolution_sync_nested_playlists_and_global_cap(self):
        from youdub import pipeline
        from youdub import step000_video_downloader as dl

        # 1. process_online_video forwards download_resolution -> target_resolution
        with tempfile.NamedTemporaryFile(suffix='.mp4', delete=False) as tf:
            tf.write(b'fake_mp4')
            local_mp4 = tf.name
        try:
            with mock.patch.object(pipeline, 'process_local_video', return_value='OK') as mock_local:
                res = pipeline.process_online_video(
                    local_mp4,
                    root_folder='out_vids',
                    download_resolution='2160p',
                    source_language='ja',
                    target_language='English',
                    tts_engine='xtts',
                    burn_subtitles=False,
                    speed_up=1.2,
                    max_workers=2,
                    xtts_cpml_ack=True,
                )
                self.assertEqual(res, 'OK')
                mock_local.assert_called_once_with(
                    local_mp4,
                    root_folder='out_vids',
                    source_language='ja',
                    target_language='English',
                    tts_engine='xtts',
                    burn_subtitles=False,
                    speed_up=1.2,
                    max_workers=2,
                    xtts_cpml_ack=True,
                    target_resolution='2160p',
                )
        finally:
            os.remove(local_mp4)

        # 2. download_single_video re-downloads when resolution changes
        with tempfile.TemporaryDirectory() as tmp:
            info = {'title': 'Demo', 'uploader': 'Ch', 'upload_date': '20250101', 'webpage_url': 'https://x.test/v1'}
            downloads = []

            class _FakeDownloadYDL:
                def __init__(self, opts):
                    self.opts = opts

                def __enter__(self):
                    return self

                def __exit__(self, *a):
                    pass

                def download(self, urls):
                    downloads.append(self.opts['format'])
                    out_dir = os.path.dirname(self.opts['outtmpl'])
                    os.makedirs(out_dir, exist_ok=True)
                    with open(os.path.join(out_dir, 'download.mp4'), 'wb') as f:
                        f.write(b'video_bytes')

            with mock.patch.object(dl.yt_dlp, 'YoutubeDL', _FakeDownloadYDL):
                f1 = dl.download_single_video(info, tmp, resolution='480p')
                self.assertIsNotNone(f1)
                self.assertEqual(len(downloads), 1)
                # Same resolution -> cached skip
                dl.download_single_video(info, tmp, resolution='480p')
                self.assertEqual(len(downloads), 1)
                # Upgraded resolution -> triggers fresh download
                dl.download_single_video(info, tmp, resolution='1080p')
                self.assertEqual(len(downloads), 2)
                self.assertIn('height<=1080', downloads[-1])

        # 3. Nested channel/tab playlists flatten cleanly and enforce global num_videos cap across URLs
        class _FakePlaylistYDL:
            def __init__(self, opts):
                self.opts = opts

            def __enter__(self):
                return self

            def __exit__(self, *a):
                pass

            def extract_info(self, u, download=False):
                # Must NOT request restrictive 'format': 'best' during metadata extraction
                assert 'format' not in self.opts
                if 'ch1' in u:
                    return {
                        'entries': [
                            {'entries': [{'id': 'v1', 'title': 'V1'}, {'id': 'v2', 'title': 'V2'}]},
                            {'entries': [{'id': 'v3', 'title': 'V3'}]},
                        ]
                    }
                return {'entries': [{'id': 'v4', 'title': 'V4'}, {'id': 'v5', 'title': 'V5'}]}

        with mock.patch.object(dl.yt_dlp, 'YoutubeDL', _FakePlaylistYDL):
            got = list(dl.get_info_list_from_url(['https://x.test/ch1', 'https://x.test/ch2'], num_videos=4))
            self.assertEqual([x['id'] for x in got], ['v1', 'v2', 'v3', 'v4'])

    def test_09_local_video_basename_collision_mtime_refresh_remux_and_comma_inputs(self):
        from youdub import pipeline

        with tempfile.TemporaryDirectory() as tmp:
            dir_a = os.path.join(tmp, 'dir_a')
            dir_b = os.path.join(tmp, 'dir_b')
            out_root = os.path.join(tmp, 'videos')
            os.makedirs(dir_a)
            os.makedirs(dir_b)

            vid_a = os.path.join(dir_a, 'same_name.mp4')
            vid_b = os.path.join(dir_b, 'same_name.mp4')
            with open(vid_a, 'wb') as f:
                f.write(b'AAA')
            with open(vid_b, 'wb') as f:
                f.write(b'BBBBBB')

            folder_a = pipeline._prepare_local_video_folder(vid_a, out_root)
            folder_b = pipeline._prepare_local_video_folder(vid_b, out_root)
            self.assertNotEqual(folder_a, folder_b)
            self.assertEqual(os.path.getsize(os.path.join(folder_a, 'download.mp4')), 3)
            self.assertEqual(os.path.getsize(os.path.join(folder_b, 'download.mp4')), 6)

            # Updating vid_a refreshes folder_a/download.mp4 instead of keeping stale copy
            with open(vid_a, 'wb') as f:
                f.write(b'AAAAAAAAA')
            os.utime(vid_a, ns=(time.time_ns() + 10_000_000, time.time_ns() + 10_000_000))
            folder_a2 = pipeline._prepare_local_video_folder(vid_a, out_root)
            self.assertEqual(folder_a2, folder_a)
            self.assertEqual(os.path.getsize(os.path.join(folder_a, 'download.mp4')), 9)

            # Non-MP4 (.mkv) uses ffmpeg remux instead of blind byte copy
            vid_mkv = os.path.join(dir_a, 'clip.mkv')
            with open(vid_mkv, 'wb') as f:
                f.write(b'MKV_RAW')

            def _fake_ffmpeg(cmd, check=True):
                dst = cmd[-2]
                with open(dst, 'wb') as wf:
                    wf.write(b'REMUXED_MP4')

            with mock.patch.object(pipeline.subprocess, 'run', side_effect=_fake_ffmpeg) as m_run:
                folder_mkv = pipeline._prepare_local_video_folder(vid_mkv, out_root)
                self.assertEqual(m_run.call_count, 1)
                with open(os.path.join(folder_mkv, 'download.mp4'), 'rb') as rf:
                    self.assertEqual(rf.read(), b'REMUXED_MP4')

            # _parse_multi_input preserves commas in existing paths and single URLs
            comma_file = os.path.join(dir_a, '你好，世界,test.mp4')
            with open(comma_file, 'wb') as f:
                f.write(b'123')
            self.assertEqual(pipeline._parse_multi_input(comma_file, is_local_paths=True), [comma_file])
            self.assertEqual(
                pipeline._parse_multi_input('https://example.com/v?a=1,2', is_local_paths=False),
                ['https://example.com/v?a=1,2'],
            )
            self.assertEqual(
                pipeline._parse_multi_input('https://a.com/1,https://b.com/2', is_local_paths=False),
                ['https://a.com/1', 'https://b.com/2'],
            )

    def test_10_whisperx_segment_sanitization_speaker_none_and_empty_transcript(self):
        from youdub import step020_whisperx as wx

        raw_segments = [
            {'text': 'Hello', 'words': [{'word': 'Hello', 'start': 0.2, 'end': 0.8}], 'speaker': None},
            {'start': None, 'end': None, 'text': 'world', 'words': [], 'speaker': ''},
            {'start': 0.9, 'end': 1.5, 'text': '   '},
        ]
        sanitized = wx._sanitize_aligned_segments(raw_segments)
        self.assertEqual(len(sanitized), 2)
        self.assertEqual(sanitized[0]['start'], 0.2)
        self.assertEqual(sanitized[0]['end'], 0.8)
        self.assertEqual(sanitized[0]['speaker'], 'SPEAKER_00')
        self.assertGreaterEqual(sanitized[1]['start'], 0.8)
        self.assertGreater(sanitized[1]['end'], sanitized[1]['start'])

        merged = wx.merge_segments(sanitized, max_gap=0.5)
        self.assertEqual(len(merged), 1)
        self.assertEqual(merged[0]['text'], 'Hello world')
        self.assertEqual(merged[0]['speaker'], 'SPEAKER_00')

        # Empty post-sanitization transcript must raise RuntimeError rather than saving []
        with tempfile.TemporaryDirectory() as tmp:
            vocals = os.path.join(tmp, 'audio_vocals.wav')
            with open(vocals, 'wb') as f:
                f.write(b'x' * 200)

            fake_model = types.SimpleNamespace(
                transcribe=lambda *a, **k: {'language': 'en', 'segments': [{'text': '   '}]}
            )
            with mock.patch.object(wx, 'load_whisper_model', return_value=fake_model), \
                 mock.patch.object(wx, 'load_align_model', return_value=(object(), {})), \
                 mock.patch.object(wx.whisperx, 'align', return_value={'segments': [{'text': '   '}]}):
                with self.assertRaises(RuntimeError):
                    wx.transcribe_audio(tmp, diarization=False, device='cpu')
            self.assertFalse(os.path.exists(os.path.join(tmp, 'transcript.json')))

    def test_11_translation_alpha_sentence_split_batch_validation_and_multi_json(self):
        from youdub import step030_translation as tr

        sents = tr.split_text_into_sentences('Dr. Smith arrived at 5 p.m. Hello world! How are you?')
        self.assertEqual(
            sents,
            ['Dr. Smith arrived at 5 p.m.', 'Hello world!', 'How are you?'],
        )

        # _parse_numbered_batch strips wrappers & rejects meta-contaminated lines
        batch_reply = (
            '1. Translation: "First clean sentence."\n'
            '2. 翻译：这句的意思是第二句\n'
            '3. "Third clean sentence."\n'
        )
        parsed = tr._parse_numbered_batch(
            batch_reply,
            expected=3,
            target_language='English',
            batch_texts=['第一句。', '第二句。', '第三句。'],
        )
        self.assertEqual(parsed[0], 'First clean sentence.')
        self.assertIsNone(parsed[1])
        self.assertEqual(parsed[2], 'Third clean sentence.')

        # _extract_json_object handles echoed empty template + real JSON object in same response
        llm_reply = (
            'Sure! Template: {"title": "", "summary": ""} '
            'Result: ```json {"title": "Real Title", "summary": "Real Summary"} ```'
        )
        extracted = tr._extract_json_object(llm_reply)
        self.assertEqual(extracted, {'title': 'Real Title', 'summary': 'Real Summary'})

    def test_12_singleflight_model_loading_and_cosyvoice_device_binding(self):
        from youdub import step010_demucs_vr as s1
        from youdub import step020_whisperx as s2
        from youdub import step042_tts_xtts as s42
        from youdub import step043_tts_cosyvoice as s43

        # 1. Demucs singleflight
        s1._separators.clear()
        s1._separator_load_locks.clear()
        demucs_calls = 0

        def _slow_sep(*a, **k):
            nonlocal demucs_calls
            demucs_calls += 1
            time.sleep(0.05)
            return object()

        with mock.patch.object(s1, 'Separator', side_effect=_slow_sep):
            with ThreadPoolExecutor(max_workers=6) as pool:
                res = list(pool.map(lambda _: s1.load_model('htdemucs_ft', device='cpu', shifts=2), range(6)))
        self.assertEqual(demucs_calls, 1)
        self.assertEqual(len({id(x) for x in res}), 1)

        # 2. WhisperX singleflight
        s2._whisper_models.clear()
        s2._whisper_load_locks.clear()
        wx_calls = 0

        def _slow_wx(*a, **k):
            nonlocal wx_calls
            wx_calls += 1
            time.sleep(0.05)
            return object()

        with mock.patch.object(s2.whisperx, 'load_model', side_effect=_slow_wx):
            with ThreadPoolExecutor(max_workers=6) as pool:
                res = list(pool.map(lambda _: s2.load_whisper_model('large-v3', device='cpu'), range(6)))
        self.assertEqual(wx_calls, 1)
        self.assertEqual(len({id(x) for x in res}), 1)

        # 3. XTTS singleflight
        s42._models.clear()
        s42._load_locks.clear()
        xtts_calls = 0

        def _slow_xtts(*a, **k):
            nonlocal xtts_calls
            xtts_calls += 1
            time.sleep(0.05)
            return object()

        with mock.patch.object(s42, '_instantiate_xtts', side_effect=_slow_xtts):
            with ThreadPoolExecutor(max_workers=6) as pool:
                res = list(pool.map(lambda _: s42.load_model('/tmp/fake_xtts', device='cpu'), range(6)))
        self.assertEqual(xtts_calls, 1)
        self.assertEqual(len({id(x) for x in res}), 1)

        # 4. CosyVoice singleflight + explicit device binding
        s43._cosyvoice_models.clear()
        s43._cosyvoice_load_locks.clear()
        cv_calls = 0

        class _SubMod:
            def __init__(self):
                self.moved_to = None

            def to(self, dev):
                self.moved_to = str(dev)
                return self

        class _FakeCosyVoice:
            def __init__(self, model_dir):
                nonlocal cv_calls
                cv_calls += 1
                time.sleep(0.05)
                self.frontend = types.SimpleNamespace(device='cuda', spk2info={})
                self.model = types.SimpleNamespace(
                    device='cuda',
                    llm=_SubMod(),
                    flow=_SubMod(),
                    hift=_SubMod(),
                    llm_context='old_cuda_stream',
                )

        fake_cv_pkg = types.ModuleType('cosyvoice')
        fake_cv_cli = types.ModuleType('cosyvoice.cli')
        fake_cv_mod = types.ModuleType('cosyvoice.cli.cosyvoice')
        fake_cv_mod.CosyVoice = _FakeCosyVoice
        with mock.patch.dict(
            sys.modules,
            {
                'cosyvoice': fake_cv_pkg,
                'cosyvoice.cli': fake_cv_cli,
                'cosyvoice.cli.cosyvoice': fake_cv_mod,
            },
        ):
            with ThreadPoolExecutor(max_workers=6) as pool:
                res = list(pool.map(lambda _: s43.load_cosyvoice_model(device='cpu'), range(6)))
        self.assertEqual(cv_calls, 1)
        loaded_cv = res[0]
        self.assertEqual(str(loaded_cv.frontend.device), 'cpu')
        self.assertEqual(str(loaded_cv.model.device), 'cpu')
        self.assertEqual(loaded_cv.model.llm.moved_to, 'cpu')
        self.assertNotEqual(loaded_cv.model.llm_context, 'old_cuda_stream')

    def test_13_app_gate_and_pipeline_language_engine_validation(self):
        import app
        from youdub import pipeline
        from youdub import step042_tts_xtts as xtts

        # Untrusted pickle global must never be allowlisted
        self.assertFalse(xtts._allowlist_unsupported_global_from_error('Unsupported global: GLOBAL os.system'))

        # Pipeline-level validation
        with self.assertRaises(ValueError):
            pipeline._validate_engine_and_language('xtts', '简体中文', xtts_cpml_ack=False)
        with self.assertRaises(ValueError):
            pipeline._validate_engine_and_language('cosyvoice', 'Deutsch', xtts_cpml_ack=False)
        pipeline._validate_engine_and_language('cosyvoice', '简体中文', xtts_cpml_ack=False)
        pipeline._validate_engine_and_language('xtts', 'Deutsch', xtts_cpml_ack=True)

        # Gradio UI gate: test exact 11 positional args (Online) & 9 positional args (Local) + kwargs
        seen_online = []
        seen_local = []

        def _spy_online(*args, **kwargs):
            seen_online.append((args, kwargs))
            return 'ONLINE_OK'

        def _spy_local(*args, **kwargs):
            seen_local.append((args, kwargs))
            return 'LOCAL_OK'

        # Give spies the exact signatures of process_online_video / process_local_video
        _spy_online.__signature__ = app.inspect.signature(pipeline.process_online_video)
        _spy_local.__signature__ = app.inspect.signature(pipeline.process_local_video)

        gated_online = app._xtts_gate(_spy_online)
        gated_local = app._xtts_gate(_spy_local)

        # Online 11 positional args with cosyvoice + 简体中文 -> succeeds
        out1 = gated_online('https://y.be/1', 'videos', '1080p', 20, 'auto', '简体中文', 'cosyvoice', True, 1.05, 1, False)
        self.assertEqual(out1, 'ONLINE_OK')

        # Online 11 positional args with cosyvoice + Français -> blocked with clear error
        out2 = gated_online('https://y.be/1', 'videos', '1080p', 20, 'auto', 'Français', 'cosyvoice', True, 1.05, 1, False)
        self.assertIn('CosyVoice-300M does not support', out2)

        # Local 9 positional args with xtts + ack=False -> blocked by CPML
        out3 = gated_local('/tmp/a.mp4', 'videos', 'auto', 'Deutsch', 'xtts', True, 1.05, 1, False)
        self.assertIn('CPML', out3)

        # Local 9 positional args with xtts + ack=True -> succeeds without duplicate-arg TypeError!
        out4 = gated_local('/tmp/a.mp4', 'videos', 'auto', 'Deutsch', 'xtts', True, 1.05, 1, True)
        self.assertEqual(out4, 'LOCAL_OK')

        # Keyword call omitting tts_engine (defaults to cosyvoice) + target_language='русский' -> blocked!
        out5 = gated_online('https://y.be/1', target_language='русский')
        self.assertIn('CosyVoice-300M does not support', out5)

    def test_14_spoken_quantifiers_cardinal_priority_and_digit_codes(self):
        from youdub.cn_tx import TextNorm
        norm = TextNorm()

        # Bug 1 verification: quantifiers like 人, 台, 把, etc. must be read as Cardinals, not digits
        self.assertEqual(norm('10000人'), '一万人')
        self.assertEqual(norm('5000台电脑'), '五千台电脑')
        self.assertEqual(norm('1000把钥匙'), '一千把钥匙')
        self.assertEqual(norm('1000'), '一千')
        self.assertEqual(norm('超过5000'), '超过五千')
        self.assertEqual(norm('3.14'), '三点一四')

        # True codes, prefixes, and leading zero identifiers must be read as digits
        self.assertEqual(norm('特工007'), '特工零零七')
        self.assertEqual(norm('他是00后球员'), '他是零零后球员')
        self.assertEqual(norm('房间号1004'), '房间号一零零四')
        self.assertEqual(norm('编号9527'), '编号九五二七')

    def test_15_bilingual_subtitle_cue_preservation_and_word_boundary_safety(self):
        from youdub.step050_synthesize_video import generate_srt, split_text

        sample_cues = [{
            'start': 1.0,
            'end': 6.5,
            'text': 'Global international cooperation is vital, because climate change affects every nation.',
            'translation': '全球国际合作至关重要，因为气候变化影响每一个国家。',
            'speaker': 'SPEAKER_00',
        }]

        # 1. generate_srt preserves exact speech-aligned timestamps and wraps lines without mid-token word truncation
        with tempfile.TemporaryDirectory() as tmp:
            srt_path = os.path.join(tmp, 'subtitles.srt')
            generate_srt(sample_cues, srt_path, speed_up=1.0)
            with open(srt_path, 'r', encoding='utf-8') as f:
                srt_content = f.read()

            self.assertIn('00:00:01,000 --> 00:00:06,500', srt_content)
            # Verify no partial words like "coopera" or "tion"
            self.assertIn('cooperation', srt_content)
            self.assertIn('international', srt_content)

        # 2. split_text word-boundary snapping guarantees complete words in both parts
        split_pieces = split_text(sample_cues)
        self.assertEqual(len(split_pieces), 2)
        piece1, piece2 = split_pieces[0], split_pieces[1]
        self.assertEqual(piece1['text'], 'Global international cooperation is')
        self.assertEqual(piece2['text'], 'vital, because climate change affects every nation.')

    def test_16_spoken_dubbing_translation_prompt_and_extra_body_fallback(self):
        from youdub import step030_translation as tr

        # Upgraded dubbing prompt includes voiceover & spoken numeral requirements
        cn_sys = tr._translation_system_content('Title: Test', '简体中文')
        self.assertIn('配音', cn_sys)
        self.assertIn('一万人', cn_sys)
        self.assertIn('五千台', cn_sys)
        self.assertIn('智能体', cn_sys)

        en_sys = tr._translation_system_content('Title: Test', 'English')
        self.assertIn('voiceover', en_sys.lower())

        # extra_body is None by default for strict OpenAI / Azure API gateway compatibility
        self.assertIsNone(tr.extra_body)


if __name__ == '__main__':
    unittest.main(verbosity=2)
