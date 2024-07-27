import shutil
import os
import sys
import subprocess
import importlib
repo_url = "https://github.com/rahulgzb/project.git"
command = ["git", "clone", repo_url]


def reload_repo():
    project_dir = 'project'
    if os.path.exists(project_dir):
        shutil.rmtree(project_dir)

    subprocess.run(command, check=True)
    sys.path.append(project_dir) 
    import project
    importlib.reload(project)

    
