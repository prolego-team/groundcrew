# Groundcrew

## Installation

Run the following commands to install `groundcrew` in a dedicated Python (Anaconda) environment.

```shell
git clone https://github.com/prolego-team/groundcrew.git
conda env create -f groundcrew/env.yaml
conda activate groundcrew
pip install -e groundcrew
cd groundcrew
```

Next, make sure you have `neo-sophia` cloned and accessible.  If you cloned both `groundcrew` and `neo-sophia` in your home directory then you are good to go.  Otherwise, open `groundcrew/config.yaml` and put the path to `neo-sophia` in the `repository` field on the first row of the file.

> If you want to ask questions about another codebase, simply change the `repository` entry in `config.yaml` to the path of that repository.

## Use Pre-Generated File Descriptions (Prolego internal use only)

The first time you run `groundcrew` on a codebase it will use an LLM to generate summaries of files in the repo.  This can take quite a bit of time.  To skip this step you can download pre-generated files to jump right into the code Q&A.

1. Download `descriptions.pkl` and `tools.yaml` from [this Google drive location](https://drive.google.com/drive/u/1/folders/16CDEMygEX9u-Kon0h-MFGoe5jTQY_Bd6).  Save these files to the `Downloads` on your Mac.
2. From the `groundcrew` directory, run the following commands.

```shell
mkdir .cache
cp ~/Downloads/descriptions.pkl .cache/
cp ~/Downloads/tools.yaml .cache/
```

## Run Groundcrew

To run `groundcrew` you will need your OpenAI API key saved as an environmental variable.  It is not already set, run `export OPENAI_API_KEY=<your private API key>` to set it.

To start the application run `TOKENIZERS_PARALLELISM=false python -m scripts.run` from the `groundcrew` directory.  After everything loads you will be presented with a user prompt.  Type your questions here and wait for the system to respond.

Example questions:

1. What does the ReACT agent do?
2. What function can load PDF files from a folder?
3. Is there functionality to call the OpenAI API?
