import ast


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


def extract_func_def_and_docstring(datapoint, language: str):
    func = datapoint["func_code_string"]
    if language == "python":
        collector = DocStringCollector()
        collector.visit(ast.parse(func))
        if collector.func_defs == None or len(collector.func_defs) == 0:
            return None, None
        func_def_parts = collector.func_defs[0]
        func_def = f"def {func_def_parts['name']}({', '.join(func_def_parts['args'])}):\n"
        docstring = "\t" + "\n\t".join(func_def_parts['docstring'].split("\n"))
        return func_def, f"""{func_def}    \"\"\"\n{docstring}\n    \"\"\""""
    elif language == "java":
        func_def = func.split('{')[0].strip()
        docstring = "\t* " + "\n\t* ".join(datapoint["func_documentation_string"].split("\n"))
        return func_def, f"""{func_def} {{\n    /**\n{docstring}\n    */"""
