import ast
import re


def remove_docstring(node):
    if isinstance(node, ast.FunctionDef) and node.body and isinstance(node.body[0], ast.Expr) and isinstance(
            node.body[0].value, ast.Str):
        node.body = node.body[1:]
    for child in ast.iter_child_nodes(node):
        remove_docstring(child)


def remove_initial_docstring(node):
    if isinstance(node, ast.FunctionDef) and node.body and isinstance(node.body[0], ast.Expr) and isinstance(
            node.body[0].value, ast.Str):
        node.body.pop(0)
    else:
        for child in ast.iter_child_nodes(node):
            remove_initial_docstring(child)


def extract_function_body(func_string, language):
    """
    It extracts the function body (or the completion) from a string containing a Java or Python function
    """
    if language == "python":
        mode = compile(func_string, '<string>', 'exec', ast.PyCF_ONLY_AST)
        remove_docstring(mode)
        body = [line for elem in mode.body[0].body for line in str(ast.unparse(elem)).split("\n")]
        code = "\t" + "\n\t".join(body)
        return code
    elif language == "java":
        return func_string.split('{', 1)[1]


"""
Extract function definition and documentation from function string for python
"""


class DocStringCollector(ast.NodeVisitor):
    def __init__(self):
        self.func_defs = []
        self.body = []

    def visit_FunctionDef(self, node):
        docstring = ast.get_docstring(node)
        if docstring is not None:
            self.func_defs.append({
                'name': node.name,
                'args': [arg.arg for arg in node.args.args],
                'docstring': docstring,
            })
        self.generic_visit(node)


def extract_code_parts_kexercises(datapoint):
    doc_pattern = r"/\*\*.*?\*/"
    doc_match = re.search(doc_pattern, datapoint["problem"], re.DOTALL)
    if doc_match:
        doc_string = doc_match.group()
    else:
        print(datapoint["problem"])
        print("--------------------------------------------------")
        return None, None, None

    # Extract the code
    fdef_string = re.sub(doc_pattern, '', datapoint["problem"], flags=re.DOTALL).strip()

    return fdef_string, doc_string, ""


def extract_code_parts_csn(datapoint, language):
    func = datapoint["func_code_string"]
    if language == "python":
        collector = DocStringCollector()
        collector.visit(ast.parse(func))
        if collector.func_defs == None or len(collector.func_defs) == 0:
            return None, None
        func_def_parts = collector.func_defs[0]
        func_def = f"def {func_def_parts['name']}({', '.join(func_def_parts['args'])}):\n"
        docstring = "\t" + "\n\t".join(func_def_parts['docstring'].split("\n"))
        return func_def, docstring, f"""{func_def}    \"\"\"\n{docstring}\n    \"\"\""""
    elif language == "java":
        func_def = func.split('{')[0].strip()
        docstring = "\t* " + "\n\t* ".join(datapoint["func_documentation_string"].split("\n"))
        return func_def, docstring, f"""{func_def} {{\n    /**\n{docstring}\n    */"""


def extract_func_def_and_docstring(datapoint, language: str):
    """
    The function extracts the function definition, docstring and prompt from the given data point.

    Parameters:
    datapoint (dict): Dictionary containing the 'func_code_string' and 'func_documentation_string'.
    language (str): The language (either "python" or "java") of the function in the datapoint.

    Returns:
    Tuple[str, str]: A tuple containing two strings.
                     The first one is the function definition
                     and the second is the function definition with the docstring
    """
    if language == "kotlin":
        return extract_code_parts_kexercises(datapoint)

    return extract_code_parts_csn(datapoint, language)
