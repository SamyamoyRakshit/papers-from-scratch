**For Image Path URL:**\
According to the spec, spaces are not allowed in URLs.So they must be encoded.

| Character | URL-encoded version |
| --------- | ------------------- |
| space     | `%20`               |
| `?`       | `%3F`               |
| `#`       | `%23`               |

Example: \
This will break in HTML:
```
<img src="Markdown Images/table.png" />
```

This works:
```
<img src="Markdown%20Images/table.png" />
```
<br>
<br>

<u> `uv` installed to: </u>

```swift
/Users/samyamoyrakshit/.local/bin
```
...but that directory isn’t in your zsh `PATH`, so your terminal can’t find `uv`.

<br>

Here are the steps to create and activate the `virtual environment`:
<br>
1. Add uv to your PATH in zsh:

    ```bash
    echo 'export PATH="$HOME/.local/bin:$PATH"' >> ~/.zshrc
    source ~/.zshrc
    ```
<br>
2. Now Check:

    ```bash
    uv --version
    ```
<br>
3. Then Install Virtual Environment:

    ```bash
    uv venv .venv-papers-replication --python 3.12
    ```
<br>
4. Activate it:

    ```bash
    source .venv-papers-replication/bin/activate
    ```