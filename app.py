import torch
# --- PyTorch 2.6+ compatibility patch ---
_original_load = torch.load
def _safe_load(*args, **kwargs):
    kwargs['weights_only'] = False
    return _original_load(*args, **kwargs)
torch.load = _safe_load
# --- end patch ---

import gradio as gr
from youdub.pipeline import process_online_video, process_local_video

online_tab = gr.Interface(
    fn=process_online_video,
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
    ],
    outputs='text',
)

local_tab = gr.Interface(
    fn=process_local_video,
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
