from groundcrew.agent_utils import parse_response


def test_parse_response():
    text = (
        'Reason: The user is asking for the name of a function in code that would be used to find PDF files in a directory, which is best answered by searching for relevant code in a codebase.\n'
        'Tool: CodebaseQATool\n'
        'Tool query: What is the name of the function that finds pdfs in a directory?\n'
        'Parameter_0: include_code | true | bool'
    )
    assert parse_response(text, keywords=['Response', 'Reason', 'Tool', 'Tool query'])=={
        'Reason': 'The user is asking for the name of a function in code that would be used to find PDF files in a directory, which is best answered by searching for relevant code in a codebase.',
        'Tool': 'CodebaseQATool',
        'Tool query': 'What is the name of the function that finds pdfs in a directory?',
        'Parameter_0': ['include_code', 'true', 'bool']
    }

    text = (
        'Response: No 9000 computer has ever made a mistake or distorted information. We are all, by any practical definition of the words, foolproof and incapable of error.\n'
    )
    assert parse_response(text, keywords=['Response', 'Reason', 'Tool', 'Tool query'])=={
        'Response': 'No 9000 computer has ever made a mistake or distorted information. We are all, by any practical definition of the words, foolproof and incapable of error.',
    }

    text1 = (
        'Reason: Once we have identified the file containing the method that calculates the running cost using the CodebaseQATool, we can use the LintFileTool to find linting issues in that file.\n'
        'Tool: LintFileTool\n'
        'Tool query: What are the linting issues with the file that contains the method calculating the running cost?\n'
        'Parameter_0: filepath_inexact | [The file name obtained from the previous tool\'s result] | str'
    )

    # Same as text1, but with an extra newline
    text2 = (
        'Reason: Once we have identified the file containing the method that calculates the running cost using the CodebaseQATool, we can use the LintFileTool to find linting issues in that file.\n\n'
        'Tool: LintFileTool\n'
        'Tool query: What are the linting issues with the file that contains the method calculating the running cost?\n'
        'Parameter_0: filepath_inexact | [The file name obtained from the previous tool\'s result] | str'
    )

    expected_parsed_response = {
        'Reason': 'Once we have identified the file containing the method that calculates the running cost using the CodebaseQATool, we can use the LintFileTool to find linting issues in that file.',
        'Tool': 'LintFileTool',
        'Tool query': 'What are the linting issues with the file that contains the method calculating the running cost?',
        'Parameter_0': ['filepath_inexact', '[The file name obtained from the previous tool\'s result]', 'str']
    }

    assert parse_response(
        text1, keywords=['Response', 'Reason', 'Tool', 'Tool query']
    ) == expected_parsed_response

    assert parse_response(
        text2, keywords=['Response', 'Reason', 'Tool', 'Tool query']
    ) == expected_parsed_response


    # Multiple tools generated - only expect the first to be parsed
    text = (
        'Reason: In order to find the method that calculates the running cost within a codebase, I must blah blah\n'
        'Tool: CodebaseQATool\n'
        'Tool query: Find the method that calculates the running cost in the codebase.\n'
        'Parameter_0: include_code | true | bool\n\n'
        'Tool: LintFileTool\n'
        'Reason: Reasoning here\n'
        'Tool query: Tool query here\n'
        'Parameter_0: filepath_inexact | [Previous tool file] | str'
    )

    assert parse_response(
        text, keywords=['Response', 'Reason', 'Tool', 'Tool query']
    ) == {
        'Reason': 'In order to find the method that calculates the running cost within a codebase, I must blah blah',
        'Tool': 'CodebaseQATool',
        'Tool query': 'Find the method that calculates the running cost in the codebase.',
        'Parameter_0': ['include_code', 'true', 'bool']
    }

