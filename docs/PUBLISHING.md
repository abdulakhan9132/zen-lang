# Publishing Guide for ZEN

Follow these steps to publish ZEN to GitHub, PyPI, and the VS Code Marketplace.

---

## 0. Check Name Availability

- **PyPI:** Visit `https://pypi.org/project/zen-lang/` (404 = available) or check from Python:
  ```bash
  python -c "import urllib.request; print(urllib.request.urlopen('https://pypi.org/pypi/zen-lang/json').status)"
  ```
- **VS Code Marketplace:** Visit `https://marketplace.visualstudio.com/items?itemName=<publisher>.zen-lang` or search `zen-lang` on https://marketplace.visualstudio.com/.

---

## 1. Push to GitHub

1. Create a new empty repository on GitHub (e.g. `zen-lang`).
2. Run from `D:\ZEN`:
   ```bash
   git init
   git add .
   git commit -m "Release ZEN v0.1.0"
   git branch -M main
   git remote add origin https://github.com/<your-username>/zen-lang.git
   git push -u origin main
   ```

---

## 2. Upload to PyPI (using `twine` + API Token)

1. Install build & upload tools:
   ```bash
   python -m pip install --upgrade build twine
   ```
2. Build the source distribution and wheel:
   ```bash
   python -m build
   python -m twine check dist/*
   ```
3. Create an API token at https://pypi.org/manage/account/token/ (scope: entire account for first upload).
4. Upload to PyPI (use `__token__` as username and paste your `pypi-...` token as password):
   ```bash
   python -m twine upload dist/*
   ```
   *(Optional: test first on TestPyPI with `python -m twine upload --repository testpypi dist/*`.)*

---

## 3. Publish VS Code Extension (`.vsix`)

1. Create a publisher on https://marketplace.visualstudio.com/manage (if your publisher ID differs from `zen-lang`, update `"publisher"` in `tools/vscode-zen/package.json` and rebuild the `.vsix`).
2. Create a Personal Access Token (PAT) in Azure DevOps with **Marketplace -> Manage** scope.
3. Package and publish from `tools/vscode-zen/`:
   ```bash
   cd tools/vscode-zen
   npx --yes @vscode/vsce package --allow-missing-repository
   npx --yes @vscode/vsce login <your-publisher>
   npx --yes @vscode/vsce publish
   ```
   Alternatively, upload the generated `.vsix` file directly on https://marketplace.visualstudio.com/manage.
