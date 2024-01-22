"""

"""


def parse_response(
        text: str,
        keywords: list[str]
    ) -> dict[str, str | list[str]]:
    """
    Parse an LLM response with sections including Reason:, Tool: and numbered
    Parameter_N lines into a dictionary of section lines

    Args:
        text (str): The text to be parsed.
        keywords (list[str]): A list of keywords to look for in the text.

    Returns:
        dict: A dictionary representation of the parsed text.
    """

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
    for keyword, keyword_lines in parsed_dict.items():

        keyword_value = '\n'.join(keyword_lines)

        if keyword.startswith('Parameter_'):
            keyword_value = keyword_value.split(' | ')

        result_dict[keyword] = keyword_value

    return result_dict
