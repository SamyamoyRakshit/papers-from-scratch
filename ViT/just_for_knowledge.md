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