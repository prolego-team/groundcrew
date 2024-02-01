"""
Default Python Template.
"""

# Copyright (c) 2023 Prolego Inc. All rights reserved.
# Ben Zimmer

from typing import Any, Dict, List, Optional, Tuple
import os
import subprocess


def main():
    """main program"""


    # res = proc.communicate('ls -l')
    # print(res[0])
    #
    # res = proc.communicate('cd ..')
    # print(res[0])

    env_child = os.environ.copy()

    while True:
        command = input('> ')

        proc = subprocess.Popen(
            # ['/bin/zsh', '-c', command],
            command,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            encoding='utf8',
            bufsize=1,
            shell=True,
            env=env_child
        )

        output, code = proc.communicate()
        print(output)


if __name__ == '__main__':
    main()
