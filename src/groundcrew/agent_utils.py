"""

"""


def parse_response(text: str, keywords: list[str]) -> dict[str, str | list[str]]:
    """
    Parse the first LLM response in a text with sections including Reason:, Tool: and numbered
    Parameter_N lines into a dictionary of section lines. Ignores any subsequent responses
    and handles accidental newlines.

    Args:
        text (str): The text to be parsed, potentially containing multiple responses.
        keywords (list[str]): A list of keywords to look for in the text.

    Returns:
        dict: A dictionary representation of the parsed text for the first response.
    """

    if '```python' in text:
        text = text.replace('```python', '').replace('```', '')

    # Split the text into lines
    lines = text.split('\n')

    parsed_dict = {}
    current_keyword = None
    current_value = []
    response_started = False
    tool_encountered = False

    for line in lines:

        # Check for the start of a new tool section
        if line.startswith('Tool:'):

            # If a new tool section is found after the first one, stop parsing
            if tool_encountered:
                break

            tool_encountered = True

        if any(line.startswith(keyword + ":") for keyword in keywords) or line.startswith('Parameter_'):
            response_started = True

            # If there is an ongoing section, save it before starting a new one
            if current_keyword is not None and line.split(': ', 1)[0] != current_keyword:
                parsed_dict[current_keyword] = '\n'.join(current_value)
                current_value = []

            # Update the current section keyword
            current_keyword = line.split(': ', 1)[0]

        # Append line to current section value
        if current_keyword is not None and response_started:
            content = line.replace(current_keyword + ': ', '', 1)
            if content:  # Avoid adding empty lines
                current_value.append(content)

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

