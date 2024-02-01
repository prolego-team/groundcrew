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
    eval_dirs = sorted(eval_dirs)

    data = [None]

    def load_eval_runs(eval_dir_path: str) -> None:
        """Load results from a file."""
        input_file_path = os.path.join(eval_dir_path, 'eval.pkl')
        if os.path.exists(input_file_path):
            with open(input_file_path, 'rb') as f:
                data[0] = pickle.load(f)
        else:
            data[0] = {}

    def update_tests() -> gr.Dropdown:
        """Update test seleection."""
        records = list(data[0].items())
        keys = [
            f'`{x}` `{y}` {z}'
            for (x, y, z), _ in records
        ]
        value = keys[0] if keys else None
        return gr.Dropdown(choices=keys, value=value, type='index')

    def update_answer(idx: int) -> tuple[str, pd.DataFrame]:
        """Update answer text box."""

        if idx is None or data[0] is None:
            res = ''
            record = {}
        else:
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

    # not sure if this is the best way to prepare the initial state but it seems to work
    load_eval_runs(eval_dirs[0])
    ans, rec = update_answer(0)

    with gr.Blocks() as demo:
        gr.Markdown('# Review Evaluation Results')

        with gr.Row():
            with gr.Column():
                eval_runs = gr.Dropdown(label='Eval Runs', choices=eval_dirs, value=eval_dirs[0], type='value')
            with gr.Column():
                tests = update_tests()

        with gr.Row():
            with gr.Column():
                info = gr.DataFrame(label='info', value=rec,  headers=None)

        with gr.Row():
            with gr.Column():
                answer = gr.Textbox(label='Answer', value=ans, interactive=False)

        eval_runs.change(
            fn=load_eval_runs,
            inputs=[eval_runs],
            outputs=[]
        ).then(
            fn=update_tests,
            inputs=[],
            outputs=[tests]
        ).then(
            fn=update_answer,
            inputs=[tests],
            outputs=[answer, info]
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

