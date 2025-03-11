import os
import subprocess
import tempfile
from pathlib import Path

from src.cf_dataset.compiler.util import CompilerOutput


def compiles_kotlin(kotlin_code: str) -> bool:
    """
    Returns a boolean indicating whether the Kotlin code compiled successfully
    """
    return __compile_kotlin_code(kotlin_code).code == 0


def __compile_kotlin_code(kotlin_function: str, add_in_class: bool = True) -> CompilerOutput:
    """
    This function compiles a given piece of Kotlin code.

    Parameters:
    kotlin_function (str): A string representation of a Kotlin function's code.
    add_in_class (bool): A flag to decide whether the provided function needs to be enclosed within a dummy Kotlin class.

    Returns:
    CompilerOutput
    """
    if add_in_class:
        kotlin_code = __add_kotlin_function_in_class(kotlin_function)
    else:
        kotlin_code = kotlin_function

    try:
        with tempfile.NamedTemporaryFile(mode='w', suffix='.kt', delete=False) as temp_file:
            temp_file.write(kotlin_code)
            temp_file_path = temp_file.name

        # Run the Kotlin compiler (kotlinc) on the temporary file

        os.environ["PATH"] += os.pathsep + os.path.expanduser("~") + "/.sdkman/candidates/kotlin/current/bin"

        result = subprocess.run(['kotlinc', temp_file_path, '-d', 'out.jar'],
                                capture_output=True, text=True)

        if result.returncode == 0:
            return CompilerOutput("Compiled Successfully", 0)
        else:
            return CompilerOutput(result.stderr, 1)  # Pass the compiler error message
    except FileNotFoundError:
        print("Kotlin compiler (kotlinc) not found. Please make sure the Kotlin compiler is installed.")
    finally:
        os.unlink(temp_file_path)
        if Path("out.jar").exists():
            os.remove("out.jar")


def __add_kotlin_function_in_class(kotlin_code: str):
    """
    Wraps a Kotlin function inside a class definition to make it compilable.
    """
    formatted_lines = "\n    ".join(kotlin_code.split('\n'))
    return f"""
class Main {{
    companion object {{
        {formatted_lines}
    }}
}}
    """


def print_not_compilable_kotlin_count(dataset_name: str):
    """
    Prints a count of Kotlin code snippets that failed to compile.
    """
    from datasets import load_dataset
    from tqdm import tqdm

    dataset = load_dataset(dataset_name, split="train", trust_remote_code=True)
    kotlin_dataset = dataset.filter(lambda x: x["language"] == "kotlin")
    print("Dataset loaded!")
    errors = 0

    with tqdm(total=len(kotlin_dataset)) as pbar:
        for datapoint in kotlin_dataset:
            try:
                __compile_kotlin_code(datapoint["func_code_string"])
            except Exception as e:
                print(f"Error: {e}")
                errors += 1
            pbar.update(1)

    print(f"Error count: {errors}")

