# Release Process

This document describes how to create and publish a new release of Starlet.

## Overview

Starlet uses automated CI/CD for releases. When you push a version tag (e.g., `v0.2.4`), GitHub Actions automatically:

1. Runs tests on Python 3.10, 3.11, 3.12
2. Builds the distribution packages
3. Runs performance benchmarks
4. Creates a GitHub release with benchmark results
5. Publishes to PyPI

## Step-by-Step Release Process

### 1. Update the Version Number

The version is defined in **one place only**: `pyproject.toml`

Edit line 3 of `pyproject.toml`:

```toml
[project]
name = "starlet"
version = "0.2.4"  # ← Change this
```

**Note:** The version in `starlet/__init__.py` is automatically read from package metadata, so you don't need to update it manually.

### 2. Commit the Version Change

```bash
git add pyproject.toml
git commit -m "Bump version to 0.2.4"
```

### 3. Create and Push the Git Tag

The tag **must** match the format `v*.*.*` (e.g., `v0.2.4`, `v1.0.0`).

```bash
# Create the tag
git tag v0.2.4

# Push the commit
git push origin master

# Push the tag (this triggers the release workflow)
git push origin v0.2.4
```

**Important:** For your fork, use `old-origin` instead:

```bash
git push old-origin master
git push old-origin v0.2.4
```

### 4. Monitor the GitHub Actions Workflow

1. Go to: https://github.com/ucr-bdlab/starlet/actions
2. Watch the "Publish to PyPI" workflow
3. The workflow includes:
   - **Test job**: Runs pytest on 3 Python versions (~2-3 min)
   - **Build job**: Creates wheel and sdist packages (~1 min)
   - **Benchmark job**: Runs performance tests (~5-10 min)
   - **Publish job**: Uploads to PyPI (~1 min)

### 5. Verify the Release

**On GitHub:**
- Release page: https://github.com/ucr-bdlab/starlet/releases
- Should show benchmark results in the release notes

**On PyPI:**
- Package page: https://pypi.org/project/starlet/
- Verify new version appears

**Locally:**
```bash
pip install --upgrade starlet
python -c "import starlet; print(starlet.__version__)"
```

## Version Numbering

Starlet follows [Semantic Versioning](https://semver.org/):

- **Major** (X.0.0): Breaking API changes
- **Minor** (0.X.0): New features, backward compatible
- **Patch** (0.0.X): Bug fixes, backward compatible

### Examples:
- `0.2.3` → `0.2.4`: Bug fix or small improvement
- `0.2.4` → `0.3.0`: New feature (e.g., new CLI command)
- `0.3.0` → `1.0.0`: Stable API, breaking changes

## Test Releases

To test the release workflow without publishing to PyPI, use a `-test` suffix:

```bash
# In pyproject.toml
version = "0.2.4-test1"

# Create tag
git tag v0.2.4-test1
git push origin v0.2.4-test1
```

This will:
- ✅ Run all tests
- ✅ Run benchmarks
- ✅ Create a GitHub pre-release
- ❌ **NOT** publish to PyPI (skipped for `-test` tags)

## What Gets Automated

### GitHub Actions Workflow (`.github/workflows/publish.yml`)

**Triggers:** Push of tag matching `v*.*.*`

**Jobs:**

1. **test** (runs in parallel for Python 3.10, 3.11, 3.12)
   ```bash
   pytest tests/ -v --cov=starlet
   ```

2. **build** (depends on test passing)
   ```bash
   python -m build
   # Creates: dist/starlet-0.2.4.tar.gz, dist/starlet-0.2.4-py3-none-any.whl
   ```

3. **benchmark** (depends on test passing)
   ```bash
   # Downloads asia_postal_codes.parquet (184K features, 1GB)
   # Runs: starlet build --input ... --num-tiles 10 --zoom 5
   # Outputs: benchmark results in release notes
   ```

4. **publish** (depends on build)
   ```bash
   # Uploads to PyPI using trusted publisher (OIDC)
   # No manual API token needed
   ```

## Troubleshooting

### Build Fails

**Error:** Tests failing
```bash
# Run tests locally first
pytest tests/ -v
```

**Error:** Import errors
```bash
# Ensure dev dependencies installed
pip install -e ".[dev]"
```

### Tag Already Exists

**Error:** `tag 'v0.2.4' already exists`
```bash
# Delete local tag
git tag -d v0.2.4

# Delete remote tag (careful!)
git push origin :refs/tags/v0.2.4

# Recreate with correct commit
git tag v0.2.4
git push origin v0.2.4
```

### PyPI Upload Fails

**Error:** Version already exists on PyPI

PyPI doesn't allow re-uploading the same version. You must:
1. Increment version: `0.2.4` → `0.2.5`
2. Create new tag: `v0.2.5`

### Wrong Remote

If you accidentally pushed to the wrong remote:

```bash
# Your fork (where you should push)
git push old-origin master
git push old-origin v0.2.4

# Upstream (only push via PR or if you have write access)
git push origin master  # Only if authorized
```

## Manual Release (Fallback)

If GitHub Actions fails, you can publish manually:

```bash
# Install build tools
pip install --upgrade build twine

# Build distributions
python -m build

# Check the build
twine check dist/starlet-0.2.4*

# Upload to PyPI (requires API token)
twine upload dist/starlet-0.2.4*
```

**Note:** Manual releases won't have automated benchmark results.

## Checklist

Before creating a release:

- [ ] All tests pass locally: `pytest tests/`
- [ ] Version updated in `pyproject.toml`
- [ ] CLAUDE.md updated (if architecture changed)
- [ ] Commit message: `"Bump version to X.Y.Z"`
- [ ] Tag created: `git tag vX.Y.Z`
- [ ] Tag pushed: `git push origin vX.Y.Z`
- [ ] GitHub Actions workflow passed
- [ ] Release appears on GitHub
- [ ] Package appears on PyPI
- [ ] Can install: `pip install starlet==X.Y.Z`

## Release Notes

GitHub releases are auto-generated with benchmark results. To add custom notes:

1. Wait for automated release to be created
2. Go to: https://github.com/ucr-bdlab/starlet/releases
3. Click "Edit" on the release
4. Add notes **above** the benchmark results
5. Save

## Example Release Timeline

```
13:00  Update version in pyproject.toml, commit, push
13:01  Create and push tag: git tag v0.2.4 && git push origin v0.2.4
13:02  GitHub Actions starts (test job)
13:05  Tests complete, build job starts
13:06  Build complete, benchmark job starts
13:15  Benchmarks complete, publish job starts
13:16  Published to PyPI ✅
13:20  GitHub release created with benchmark results ✅
```

Total time: ~15-20 minutes

## Support

- **GitHub Actions logs**: https://github.com/ucr-bdlab/starlet/actions
- **PyPI project**: https://pypi.org/project/starlet/
- **Issues**: https://github.com/ucr-bdlab/starlet/issues
