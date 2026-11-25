# ZEN Language Support for VS Code

Provides syntax highlighting, comment toggling, bracket matching, auto-closing pairs, and the `.zen` file icon for ZEN.

## Installation

Copy this folder (`tools/vscode-zen/`) to your VS Code extensions directory:

- **Windows (PowerShell):**
  ```powershell
  Copy-Item -Recurse -Force tools\vscode-zen "$env:USERPROFILE\.vscode\extensions\zen-lang-0.1.0"
  ```
- **macOS / Linux:**
  ```bash
  cp -r tools/vscode-zen ~/.vscode/extensions/zen-lang-0.1.0
  ```

Then restart VS Code (or run **Developer: Reload Window** from the Command Palette).
