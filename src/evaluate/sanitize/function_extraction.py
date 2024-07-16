def extract_function_completion(model_output, prompt):
    prompt_end_index = model_output.find(prompt) + len(prompt)

    completion = model_output[prompt_end_index:]

    completion_lines = completion.splitlines()

    code_lines = []
    for line in completion_lines:
        if line == "": continue
        if not line.startswith('\t') and not line.startswith(' '):
            break
        code_lines.append(line)

    final_completion = "\n".join(code_lines)

    return final_completion


def normalize_indentation(prompt, completion, indent_size):
    prompt_lines = len([line.lstrip() for line in prompt.splitlines() if line.lstrip()])
    lines = (prompt + completion).strip().split('\n')

    min_indent = float('inf')
    for line in lines:
        stripped_line = line.lstrip()
        if stripped_line:
            leading_spaces = len(line) - len(stripped_line)
            if leading_spaces < min_indent:
                min_indent = leading_spaces

    normalized_lines = []
    for line in lines:
        stripped_line = line.lstrip()
        if stripped_line:
            normalized_line = ' ' * indent_size + stripped_line
            normalized_lines.append(normalized_line)
        else:
            normalized_lines.append('')

    return '\n'.join(normalized_lines[prompt_lines:]), "\n".join(normalized_lines[:prompt_lines])

