from src.cf_dataset.compiler.java_compiler import compiles
from src.cf_dataset.compiler.python_interpreter import compile_code

compile_function = {
    "java": compiles,
    "python": compile_code,
}
