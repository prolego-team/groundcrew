"""
Review evaluation results interactively.
"""

import os
import pickle

import gradio as gr
import pandas as pd


def main():
    """Setup and run gr app."""

    eval_dirs = os.listdir('.')
    eval_dirs = [
        x for x in eval_dirs
        if os.path.isdir(x) and x.startswith('eval_')
    ]

    data = [None]

    def load_eval_runs(eval_dir_path: str) -> None:
        """Load results from a file."""
        input_file_path = os.path.join(eval_dir_path, 'eval.pkl')
        with open(input_file_path, 'rb') as f:
            data[0] = pickle.load(f)

    def update_tests() -> gr.Dropdown:
        """Update test seleection."""
        records = list(data[0].items())
        keys = [
            f'`{x}` `{y}` {z}'
            for (x, y, z), _ in records
        ]
        return gr.Dropdown(choices=keys, value=0, type='index')

    def update_answer(idx: int) -> tuple[str, pd.DataFrame]:
        """Update answer text box."""
        records = list(data[0].items())
        key, record = records[idx]
        record = dict(record)
        res = record['answer']
        del record['answer']
        record = dict(
            suite=key[0],
            test=key[1],
            run=key[2],
            **record
        )

        record = pd.DataFrame({k: [v] for k, v in record.items()})
        return res, record

    with gr.Blocks() as demo:
        gr.Markdown('# Review Evaluation Results')

        with gr.Row():
            with gr.Column():
                eval_runs = gr.Dropdown(label='Eval Runs', choices=eval_dirs, value=0, type='value')
            with gr.Column():
                tests = gr.Dropdown(label='Tests', choices=[], value=0, type='index')

        with gr.Row():
            with gr.Column():
                info = gr.DataFrame(label='info', value={},  headers=None)

        with gr.Row():
            with gr.Column():
                answer = gr.Textbox(label='Answer', value='', interactive=False)

        eval_runs.change(
            fn=load_eval_runs,
            inputs=[eval_runs],
            outputs=[]
        ).then(
            fn=update_tests,
            inputs=[],
            outputs=[tests]
        )

        tests.input(
            fn=update_answer,
            inputs=[tests],
            outputs=[answer, info]
        )

    demo.queue()
    demo.launch()


if __name__ == '__main__':
    main()

