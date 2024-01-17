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