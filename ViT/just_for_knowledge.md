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

<u>Here are the steps to create and activate the `virtual environment`:</u>
1. <u>Add uv to your PATH in zsh:</u>

    ```bash
    echo 'export PATH="$HOME/.local/bin:$PATH"' >> ~/.zshrc
    source ~/.zshrc
    ```

2. <u>Now Check:</u>

    ```bash
    uv --version
    ```

3. <u>Then Install Virtual Environment:</u>

    ```bash
    uv venv .venv-papers-replication --python 3.12
    ```

4. <u>Activate it:</u>
    ```bash
    source .venv-papers-replication/bin/activate
    ```