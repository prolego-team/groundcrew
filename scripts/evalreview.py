"""
Example of using an LLM to chat with a database.
"""

from typing import Any
import os
import pickle

import gradio as gr
import pandas as pd


def main():
    """Setup and run gradio app."""

    input_file_path = os.path.join('eval_20240129_183440', 'eval.pkl')

    with open(input_file_path, 'rb') as f:
        data = pickle.load(f)

    records = list(data.items())
    keys = [
        f'`{x}` `{y}` {z}'
        for (x, y, z), _ in records
    ]

    headers = dict(records[0][1])
    del headers['answer']
    headers = list(headers.keys())

    def update(idx: int) -> Any:
        """update"""
        _, record = records[idx]
        record = dict(record)
        res = record['answer']
        del record['answer']
        record = pd.DataFrame({k: [v] for k, v in record.items()})
        return res, record

    with gr.Blocks() as demo:
        gr.Markdown('# Review Evaluation Results')

        with gr.Row():
            with gr.Column():
                tests = gr.Dropdown(choices=keys, value=0, type='index')
            with gr.Column():
                info = gr.DataFrame(value={}, headers=headers)

        with gr.Row():
            with gr.Column():
                answer = gr.Textbox(value='', label='Answer', interactive=False)

        tests.change(
            fn=update,
            inputs=[tests],
            outputs=[answer, info]
        )

    demo.queue()
    demo.launch()


if __name__ == '__main__':
    main()

