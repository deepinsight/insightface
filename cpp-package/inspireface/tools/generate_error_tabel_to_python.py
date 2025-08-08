import click
import re
from datetime import datetime

# Function to calculate the actual error code value based on the expressions
def calculate_error_code_value(error_code_str, error_definitions):
    try:
        # Replace the hex values and error definitions with actual values
        error_code_str = re.sub(r'0X([0-9A-F]+)', lambda m: str(int(m.group(1), 16)), error_code_str)
        for error_name, error_value in error_definitions.items():
            error_code_str = error_code_str.replace(error_name, str(error_value))
        # Evaluate the expression to get the actual error code value
        return eval(error_code_str)
    except Exception as e:
        # If there is an error in evaluation (like undefined reference), return the original string
        return error_code_str

# Define a function to parse the error codes between anchors in a given C header file content
def parse_and_calculate_error_codes(header_content):
    # Define the start and end anchor tags
    start_anchor = '// [Anchor-Begin]'
    end_anchor = '// [Anchor-End]'

    # Find the index of start and end anchor tags
    start_index = header_content.find(start_anchor)
    end_index = header_content.find(end_anchor, start_index)

    # Extract the content between anchors
    anchor_content = header_content[start_index+len(start_anchor):end_index].strip()

    # Split the content into lines
    lines = anchor_content.split('\n')

    # Dictionary to hold the error names and their evaluated values
    error_definitions = {}
    # List to hold the parsed error codes
    error_codes = []

    # Process each line
    for line in lines:
        line = line.strip()
        # Skip empty lines and lines that don't start with #define
        if not line or not line.startswith('#define'):
            continue
            
        # Split by comment marker and handle cases with no comment
        parts = line.split('//')
        define_part = parts[0].strip()
        comment_part = parts[1].strip() if len(parts) > 1 else ''

        # Extract the error name and code, handling parentheses
        define_parts = define_part.split(None, 2)  # Split into max 3 parts
        if len(define_parts) >= 2:  # Changed from == 3
            error_name = define_parts[1]
            error_code_str = define_parts[2].strip('()') if len(define_parts) > 2 else ''

            # Calculate the actual error code value
            error_code_value = calculate_error_code_value(error_code_str, error_definitions)

            # Store the calculated error code value for later reference
            error_definitions[error_name] = error_code_value

            # Skip base offset definitions (ending with _BASE)
            if error_name.endswith('_BASE'):
                continue

            # Append the extracted information to the error_codes list
            error_codes.append((error_name, error_code_value, comment_part))

    return error_codes

# Click command for processing the header file and outputting Python error table
@click.command()
@click.argument('header_path', type=click.Path(exists=True, file_okay=True, dir_okay=False, readable=True))
@click.argument('output_path', type=click.Path(file_okay=True, dir_okay=False, writable=True))
def process_header(header_path, output_path):
    # Read the header file content
    with open(header_path, 'r', encoding='utf-8') as file:
        header_content = file.read()

    # Parse and calculate the error codes from the header content
    parsed_error_codes = parse_and_calculate_error_codes(header_content)

    # Generate Python file content
    python_content = f'''"""
InspireFace Error Codes

Auto-generated error code definitions from {header_path}
Generated on: {datetime.now().strftime("%Y-%m-%d")}
"""

'''

    # Generate simple error code definitions
    for name, code, comment in parsed_error_codes:
        if comment:
            python_content += f"{name} = {code}  # {comment}\n"
        else:
            python_content += f"{name} = {code}\n"

    # Write the Python file
    with open(output_path, 'w', encoding='utf-8') as py_file:
        py_file.write(python_content)

    click.echo(f"Python error table has been written to {output_path}")
    click.echo(f"Generated {len(parsed_error_codes)} error code definitions")

if __name__ == '__main__':
    process_header()
