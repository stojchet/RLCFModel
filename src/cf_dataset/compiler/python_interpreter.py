def compile_code(code_string):
    try:
        compile(code_string, '<string>', 'exec')
        return True
    except SyntaxError or Exception:
        return False
    except ValueError:
        return False
