import re
import string

def clean_names_in_json(input_filename, output_filename):
    """
    Cleans the "name" fields in a JSON file by replacing non-ASCII characters with 'x' 
    and removing internal quotation marks.

    Example of problematic input:
        {
            "name": "@"�sP(0): flat_tensor"
        }
    """
    with open(input_filename, 'r', encoding='utf-8', errors='replace') as file:
        content = file.read()

        # Decode Unicode escape sequences
        content = content.encode().decode('unicode_escape')

        # Regex to find "name": "<value>"
        def replace_non_ascii_and_quotes(match):
            name = match.group(1)
            visible_printable = ''.join(c for c in string.printable if c not in '\t\n\r\x0b\x0c}{')
            cleaned_name = ''.join(c if c in visible_printable else 'x' for c in name)
            cleaned_name = cleaned_name.replace('"', 'y')  # Replace internal quotes
            return f'"name": "{cleaned_name}"'
        
        # Apply regex to clean names
        cleaned_content = re.sub(r'"name": "([\s\S]*?)"(?=, |\}|\s*\})', replace_non_ascii_and_quotes, content, flags=re.DOTALL)

    # Write the cleaned JSON data to a new file
    with open(output_filename, 'w', encoding='utf-8') as outfile:
        outfile.write(cleaned_content)

clean_names_in_json('./trace.json', './trace_new.json')