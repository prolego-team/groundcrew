"""

"""


def parse_response(text: str, keywords: list[str]) -> dict[str, str]:
    """
    Parse the provided text into a dictionary.

    Args:
        text (str): The text to be parsed.

    Returns:
        dict: A dictionary representation of the parsed text.
    """

    keywords = ['Reason', 'Tool']

    # Split the text into lines
    lines = text.split('\n')

    parsed_dict = {}
    for line in lines:

        line_start = line.split(': ', 1)[0]

        current_keyword = None
        if line_start in keywords or line_start.startswith('Parameter_'):
            current_keyword = line_start
            line = ''.join(line.split(current_keyword + ': ')[1:])

        if current_keyword is not None:
            values = parsed_dict.setdefault(current_keyword, [])
            values.append(line)

    result_dict = {}
    for key, val in parsed_dict.items():

        val = '\n'.join(val)

        if key.startswith('Parameter_'):
            val = val.split(' | ')

        result_dict[key] = val

    return result_dict
