import sys
import re
import os

def clean_file(filepath):
    if not os.path.exists(filepath):
        print(f"File not found: {filepath}")
        return

    with open(filepath, 'r', encoding='utf-8') as f:
        content = f.read()

    # Regex matches:
    # 1. Hidden unicode markers (e.g. \ue200cite...\ue201) often used by AI assistants
    # 2. Or standard space optionally followed by 'cite' and then turn/search/view
    pattern = re.compile(r' ?\ue200cite[^\ue201]*\ue201| ?cite(?:turn\d+|search\d+|view\d+)+')
    new_content = pattern.sub('', content)

    # If the file hasn't changed, no need to write
    if content == new_content:
        print(f"No citations found in {filepath}")
        return

    with open(filepath, 'w', encoding='utf-8') as f:
        f.write(new_content)
    
    print(f"Successfully cleaned citations from {filepath}")

def main():
    if len(sys.argv) < 2:
        print("Usage: python clean_cites.py <file_path1> [file_path2 ...]")
        return
    
    for filepath in sys.argv[1:]:
        clean_file(filepath)

if __name__ == '__main__':
    main()
