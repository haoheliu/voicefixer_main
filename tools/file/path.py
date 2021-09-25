import os
import git

def find_and_build(root,path):
    path = os.path.join(root, path)
    if not os.path.exists(path):
        os.makedirs(path,exist_ok=True)
    return path

def get_git_root():
    git_repo = git.Repo("", search_parent_directories=True)
    git_root = git_repo.git.rev_parse("--show-toplevel")
    return git_root
