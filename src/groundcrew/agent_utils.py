"""

"""


def parse_response(
        text: str, keywords: list[str]) -> dict[str, str | list[str]]:
    """
    Parse an LLM response with sections including Reason:, Tool: and numbered
    Parameter_N lines into a dictionary of section lines

    Args:
        text (str): The text to be parsed.
        keywords (list[str]): A list of keywords to look for in the text.

    Returns:
        dict: A dictionary representation of the parsed text.
    """

    if '```python' in text:
        text = text.replace('```python', '').replace('```', '')

    # Split the text into lines
    lines = text.split('\n')

    parsed_dict = {}
    current_keyword = None
    current_value = []

    for line in lines:

        # Check if the line is a new section
        if any(line.startswith(keyword + ":") for keyword in keywords) or line.startswith('Parameter_'):
            # If there is an ongoing section, save it before starting a new one
            if current_keyword is not None:
                parsed_dict[current_keyword] = '\n'.join(current_value)
                current_value = []

            # Update the current section keyword
            current_keyword = line.split(': ', 1)[0]

        # Append line to current section value
        if current_keyword is not None:
            current_value.append(line.replace(current_keyword + ': ', '', 1))

    # Add the last section to the dictionary
    if current_keyword is not None:
        parsed_dict[current_keyword] = '\n'.join(current_value)

    # Process Parameter_N values
    result_dict = {}
    for keyword, value in parsed_dict.items():
        if keyword.startswith('Parameter_'):
            sections = value.split(' | ')
            result_dict[keyword] = sections
        else:
            result_dict[keyword] = value

    return result_dict


text = """
Tool: SingleDocstringTool
Tool query: Based on the provided refactored function's code snippet, generate an appropriate docstring for the method `get_running_cost` which has now been updated to return a list instead of a dictionary.
Parameter_0: code | ```python
def get_running_cost(self) -> List[float]:
    Your Docstring Will Be Here

    return [self.input_cost, self.output_cost, self.input_cost + self.output_cost]
``` | str
Parameter_1: filename | none | str
Parameter_2: function_name | none | str
"""

keywords = ['Response', 'Reason', 'Tool', 'Tool query']

#print(parse_response(text, keywords))
