import os
import re

input_dir = "/data1/tianang/Projects/Intern-S1/playbooks/Josh_origin"
output_dir = "/data1/tianang/Projects/Intern-S1/playbooks/Josh_updatable"

os.makedirs(output_dir, exist_ok=True)

def convert_playbook(input_path, output_path, task_name):
    with open(input_path, 'r') as f:
        lines = f.readlines()

    output_lines = []
    
    # We will put everything under a general section initially, 
    # except if we find section headers in the original text, we can convert them.
    current_section = "GENERAL GUIDELINES"
    output_lines.append(f"## {current_section}\n")
    
    bullet_idx = 1
    
    # Get a slug for the task name to use in the ID
    slug = task_name[:3].lower()

    for line in lines:
        line = line.strip()
        if not line:
            continue
            
        if line.startswith('#'):
            # Convert to a section header
            # Remove all leading # and spaces
            section_name = re.sub(r'^#+\s*', '', line).upper()
            if section_name:
                current_section = section_name
                output_lines.append(f"\n## {current_section}\n")
        elif line.startswith('-'):
            # It's a bullet point
            content = line[1:].strip()
            if content:
                bullet_id = f"{slug}-{bullet_idx:05d}"
                output_lines.append(f"[{bullet_id}] helpful=0 harmful=0 :: {content}\n")
                bullet_idx += 1
        else:
            # Regular text line, treat as a bullet as well to preserve all info
            content = line
            bullet_id = f"{slug}-{bullet_idx:05d}"
            output_lines.append(f"[{bullet_id}] helpful=0 harmful=0 :: {content}\n")
            bullet_idx += 1

    with open(output_path, 'w') as f:
        f.writelines(output_lines)

for filename in os.listdir(input_dir):
    if filename.endswith(".txt"):
        task_name = filename[:-4]
        input_path = os.path.join(input_dir, filename)
        output_path = os.path.join(output_dir, filename)
        convert_playbook(input_path, output_path, task_name)
        print(f"Converted {filename}")

