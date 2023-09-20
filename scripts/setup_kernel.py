import os
from ipykernel.kernelspec import install_kernel_spec
from IPython.utils.tempdir import TemporaryDirectory

def install_kernel(env_name, display_name):
    with TemporaryDirectory() as td:
        os.makedirs(os.path.join(td, 'kernels', env_name))
        with open(os.path.join(td, 'kernels', env_name, 'kernel.json'), 'w') as f:
            f.write(kernel_json.format(env_name=env_name, display_name=display_name))
        install_kernel_spec(td, env_name, user=True, replace=True, prefix=None)

kernel_json = """\
{{
 "argv": ["python", "-m", "ipykernel_launcher", "-f", "{{connection_file}}"],
 "display_name": "{display_name}",
 "language": "python",
 "env": {{"PYTHONPATH": "{env_name}"}}
}}
"""

if __name__ == "__main__":
    env_name = os.environ.get('CONDA_DEFAULT_ENV')
    if env_name:
        install_kernel(env_name, f"Python ({env_name})")
    else:
        print("No conda environment is activated.")
        
    """
    In the above script:

    We import the os module to work with the operating system and get environment variables.
    We get the name of the currently activated conda environment using os.environ.get('CONDA_DEFAULT_ENV').
    We check if env_name is not None (which would indicate that a conda environment is activated) before calling the install_kernel function.
    If no conda environment is activated, a message is printed to inform the user.
    
    """

