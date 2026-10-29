import inspect

import gradio as gr
from youdub.pipeline import process_online_video, process_local_video


def _xtts_gate(process_fn):
    """Gate calls to *process_fn* behind an XTTS/CPML acknowledgment.

    The Gradio ``inputs`` list MUST place the CPML checkbox last; this wrapper
    reads ``args[-1]`` for that boolean, locates ``tts_engine`` by inspecting
    the wrapped function's signature (so the order of pipeline parameters is
    irrelevant), and refuses to invoke ``process_fn`` with ``tts_engine ==
    'xtts'`` unless the user has acknowledged CPML. The acknowledgment flag is
    then forwarded to the pipeline for defence-in-depth.
    """
    sig = inspect.signature(process_fn)
    param_names = list(sig.parameters.keys())
    tts_idx = param_names.index('tts_engine') if 'tts_engine' in param_names else None

    def wrapped(*args, **kwargs):
        ack = kwargs.pop('xtts_cpml_ack', None)
        if ack is None and args:
            ack = args[-1]
            args = args[:-1]

        tts_engine = None
        if tts_idx is not None and tts_idx < len(args):
            tts_engine = args[tts_idx]

        if tts_engine == 'xtts' and not bool(ack):
            return (
                "ERROR: XTTS v2 requires explicit CPML non-commercial acknowledgment.\n"
                "Tick the acknowledgment checkbox, or switch to 'cosyvoice' (Apache 2.0).\n"
                "See https://coqui.ai/cpml"
            )
        kwargs['xtts_cpml_ack'] = bool(ack)
        return process_fn(*args, **kwargs)

    return wrapped


online_tab = gr.Interface(
    fn=_xtts_gate(process_online_video),
    inputs=[
        gr.Textbox(label='Video URL', placeholder='YouTube / playlist / channel URL', value=''),
        gr.Textbox(label='Root Folder', value='videos'),
        gr.Radio(['1080p', '2160p', '720p', '480p'], label='Download Resolution', value='1080p'),
        gr.Slider(1, 100, step=1, label='Max Videos', value=20),
        gr.Dropdown(
            ['auto', 'en', 'zh', 'ja', 'ko', 'fr', 'de', 'ru', 'es'],
            label='Source Language', value='auto'),
        gr.Dropdown(
            ['简体中文', '繁体中文', 'English', 'Deutsch', 'Français', 'русский'],
            label='Target Language', value='简体中文'),
        gr.Dropdown(['cosyvoice', 'xtts'], label='TTS Engine', value='xtts'),
        gr.Checkbox(label='Burn Subtitles', value=True),
        gr.Slider(0.5, 2.0, step=0.05, label='Speed Up', value=1.05),
        gr.Slider(1, 8, step=1, label='Parallel Workers', value=1),
        gr.Checkbox(
            label='I acknowledge XTTS v2 is licensed under CPML (non-commercial only)',
            value=False,
        ),
    ],
    outputs='text',
)

local_tab = gr.Interface(
    fn=_xtts_gate(process_local_video),
    inputs=[
        gr.Textbox(label='Local Video Path(s)', placeholder='/abs/path/to/video1.mp4,/abs/path/to/video2.mp4', value=''),
        gr.Textbox(label='Root Folder', value='videos'),
        gr.Dropdown(
            ['auto', 'en', 'zh', 'ja', 'ko', 'fr', 'de', 'ru', 'es'],
            label='Source Language', value='auto'),
        gr.Dropdown(
            ['简体中文', '繁体中文', 'English', 'Deutsch', 'Français', 'русский'],
            label='Target Language', value='简体中文'),
        gr.Dropdown(['cosyvoice', 'xtts'], label='TTS Engine', value='cosyvoice'),
        gr.Checkbox(label='Burn Subtitles', value=True),
        gr.Slider(0.5, 2.0, step=0.05, label='Speed Up', value=1.05),
        gr.Slider(1, 8, step=1, label='Parallel Workers', value=1),
        gr.Checkbox(
            label='I acknowledge XTTS v2 is licensed under CPML (non-commercial only)',
            value=False,
        ),
    ],
    outputs='text',
)

app = gr.TabbedInterface(
    interface_list=[online_tab, local_tab],
    tab_names=['Online Video', 'Local Video'],
    title='OmniDub Studio',
)

if __name__ == '__main__':
    app.launch()
