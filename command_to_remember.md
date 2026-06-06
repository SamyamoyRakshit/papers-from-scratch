# NOTE:
#### - `uv.lock` is the authoritative source of exact package versions.
#### - `pyproject.toml` defines the project and its dependencies (similar to package.json in JS).

1. Initialize project (if not already)
uv init

2. Add dependencies to pyproject.toml (manually or using this)
uv pip add <package-name>

    Example:
    uv pip add torch torchvision matplotlib ipykernel tqdm requests numpy torchinfo

3. Compile lock file from pyproject.toml
uv pip compile pyproject.toml --output-file uv.lock

4. Install dependencies from lock file
uv pip install --requirements uv.lock

5. Update lock file when pyproject.toml changes
uv pip compile pyproject.toml --output-file uv.lock

6. Sync virtual environment to match lock file
uv pip sync --requirements uv.lock

7. Export current environment to requirements.txt (not needed if using uv.lock)
uv pip freeze > requirements.txt

8. Add the dependencies directly to 'pyproject.toml'
uv add -r requirements.txt

9. Important: Do a uv sync (use the below command as it is)
uv sync --active

#### for #9: Delete the `uv.lock` file if it leads to any versioning issues. Then run again code of #9.