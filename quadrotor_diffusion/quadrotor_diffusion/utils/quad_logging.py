from dataclasses import fields
import os
import inspect

old_print = print


def iprint(*messages):
    caller_frame = inspect.stack()[1]
    caller_filename = os.path.basename(caller_frame.filename)
    caller_line = caller_frame.lineno
    message = " ".join(map(str, messages))

    if os.getenv('DEBUG') == 'True':
        print(f"[ {caller_filename}:{caller_line} ] {message}")
    else:
        print(f"[ {caller_filename.replace('.py', '')} ] {message}")


def dataclass_to_table(data, title=None):
    """
    Converts a dataclass instance to a formatted table string with borders 
    and instance class name as title.

    - data: Dataclass to use
    - title: Title for table (default use class name)
    """
    if not hasattr(data, '__dataclass_fields__'):
        raise TypeError("Provided object is not a dataclass instance.")

    # Get class name for title
    class_name = data.__class__.__name__ if title is None else title

    # Calculate max widths
    field_width = max(len(field.name) for field in fields(data))
    value_width = max(len(str(getattr(data, field.name))) for field in fields(data))

    # Ensure minimum widths
    field_width = max(field_width, len("Field"))
    value_width = max(value_width, len("Value"))

    # Calculate total width including borders and padding
    total_width = field_width + value_width + 7  # 7 accounts for borders and padding

    # Build table
    result = []
    result.append(f"\n {class_name} ")
    result.append("╔" + "═" * (field_width + 2) + "╦" + "═" * (value_width + 2) + "╗")
    result.append(f"║ {'Field':<{field_width}} ║ {'Value':<{value_width}} ║")
    result.append("╠" + "═" * (field_width + 2) + "╬" + "═" * (value_width + 2) + "╣")

    for field in fields(data):
        name = field.name
        value = str(getattr(data, name))
        result.append(f"║ {name:<{field_width}} ║ {value:<{value_width}} ║")

    result.append("╚" + "═" * (field_width + 2) + "╩" + "═" * (value_width + 2) + "╝")

    return "\n".join(result)
