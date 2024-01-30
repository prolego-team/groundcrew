"""
Example of using an LLM to chat with a database.
"""

import sys
import logging
import sqlite3
from datetime import datetime


import os
import pickle

import gradio as gr


def main():
    """Setup and run gradio app."""

    input_file_path = os.path.join('eval_20240129_183440', 'eval.pkl')

    with open(input_file_path, 'rb') as f:
        data = pickle.load(f)

    items = list(data.items())
    keys = [
        f'`{x}` `{y}` {z}'
        for (x, y, z), _ in items
    ]

    print(keys)

    # def answer_wrapper(chat_history):
    #     final_message = chat_history[-1][1]
    #     if final_message is None:
    #         response = 'It appears the agent couldn\'t answer the questions.'
    #     elif 'Final Answer:' in final_message:
    #         response = final_message.split('Final Answer:')[1].strip()
    #         response = response.split('\n')[0].strip()
    #     else:
    #         response = 'It appears the agent couldn\'t answer the questions.'
    #
    #     return response


    def update(test_description) -> str:
        """adlfkjahdfljka"""
        print(test_description)


    with gr.Blocks() as demo:
        gr.Markdown('# Review Evaluation Results')

        # question = gr.Textbox(
        #     value=DEFAULT_QUESTION, label='Ask a question about the data')

        with gr.Row():
            with gr.Column():
                tests = gr.Dropdown(choices=keys, value=0)
            with gr.Column():
                answer = gr.Textbox(value='', label='Answer', interactive=False)

        tests.change(
            fn=update,
            inputs=[tests],
            outputs=[answer]
        )

    demo.queue()
    demo.launch()


if __name__=='__main__':
    main()