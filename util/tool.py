

def load_html_file(filepath):
    with open(filepath, "r", encoding="utf-8") as file:
        html_content = file.read()
    return html_content