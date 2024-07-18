from dataclasses import dataclass
import subprocess
import tempfile
import os

from datasets import load_dataset
from tqdm import tqdm


def compiles(prediction: str) -> bool:
    """
    Returns a boolean indicating whether the code compiled successfully
    """
    return compile_java_code(prediction).code == 0


@dataclass
class CompilerOutput:
    """
    Dataclasses used to store compilation code and output
    """
    output: str
    code: int


def compile_java_code(java_function, add_in_class: bool = True) -> CompilerOutput:
    """
    This function compiles a given piece of Java code.

    Parameters:
    java_function (str): A string representation of a java function's code.
    add_in_class (bool): A flag to decide whether the provided function needs to be enclosed within a dummy java class.

    Returns:
    CompilerOutput
    """

    if add_in_class:
        java_code = add_java_function_in_class(java_function)
    else:
        java_code = java_function
    try:
        with tempfile.NamedTemporaryFile(mode='w', suffix='.java', delete=False) as temp_file:
            temp_file.write(java_code)
            temp_file_path = temp_file.name

        result = subprocess.run(['javac', temp_file_path], capture_output=True, text=True)

        if result.returncode == 0:
            return CompilerOutput("Compiled Successfully", 0)
        else:
            if "error: cannot find symbol" in result.stderr:
                return CompilerOutput("Compiled Successfully", 0)
            return CompilerOutput(result.stderr, 1)  # Compiler error message
    except FileNotFoundError:
        print("Java compiler (javac) not found. Please make sure Java Development Kit (JDK) is installed.")
    finally:
        os.unlink(temp_file_path)


def add_java_function_in_class(java_code: str):
    formatted_lines = "\n\t".join(java_code.split('\n'))
    return f"""
class Main {{
    {formatted_lines}
}}
    """


def compile_dataset():
    dataset = load_dataset("stojchet/6K_java_base", split="train", trust_remote_code=True)
    java_dataset = dataset.filter(lambda x: x["language"] == "java")
    print("dataset loaded !")
    errors = 0
    with tqdm(total=len(java_dataset)) as pbar:
        for datapoint in java_dataset:
            try:
                compile_java_code(datapoint["func_code_string"])
            except:
                print("error")
                errors += 1
            pbar.update(1)

    print(f"Error count: {errors}")
