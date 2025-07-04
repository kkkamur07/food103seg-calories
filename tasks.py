from invoke import task


@task
def setup(ctx):
    """Set up the development environment"""
    print("Setting up development environment...")
    ctx.run("pip install -r requirements.txt")
    ctx.run("pre-commit install")
    print("Setup complete!")


@task
def clean(ctx):
    """Clean up temporary files and caches"""
    print("Cleaning up...")
    ctx.run("find . -type f -name '*.pyc' -delete")
    ctx.run("find . -type d -name '__pycache__' -delete")
    ctx.run("find . -type d -name '.pytest_cache' -delete")
    print("Cleanup complete!")


@task
def download_data(ctx):
    """Download datasets using gdown"""
    print("Downloading datasets...")
    # Add your Google Drive file IDs here
    # ctx.run("gdown https://drive.google.com/uc?id=YOUR_FILE_ID")


@task
def git(ctx, add, commit):
    """Add files, commit with message, and push to remote"""
    # Get current branch
    branch = ctx.run("git branch --show-current", hide=True).stdout.strip()

    # Git operations
    ctx.run(f"git add {add}")
    ctx.run(f"git commit -m '{commit}'")
    ctx.run(f"git push origin {branch}")

    print(f"✅ Pushed '{commit}' to branch: {branch}")
