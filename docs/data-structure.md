# Data Organization

All scripts now organize outputs by resume name in the `data/` directory.

## Directory Structure

```
/resumes/                      # Resume files (not tracked in git)
  ebony_moore.txt
  john_doe.txt
  ...

/data/                         # Output artifacts (not tracked in git)
  ebony_moore/
    extracted_skills.json
    sorted_skills.json
    task_skill_matches.json
  john_doe/
    extracted_skills.json
    sorted_skills.json
    task_skill_matches.json
```

## Usage

### 1. Extract Skills
```bash
python extract.py --resume resumes/ebony_moore.txt
# Outputs to: data/ebony_moore/extracted_skills.json
```

### 2. Sort Skills
```bash
python sort.py --resume resumes/ebony_moore.txt
# Reads from: data/ebony_moore/extracted_skills.json
# Outputs to: data/ebony_moore/sorted_skills.json
```

### 3. Match Skills to Task
```bash
python taskmatch.py --resume resumes/ebony_moore.txt
# Reads from: data/ebony_moore/extracted_skills.json
# Outputs to: data/ebony_moore/task_skill_matches.json
```

## Custom Paths

You can still override the default paths if needed:

```bash
# Extract with custom output
python extract.py --resume resumes/custom.txt --output my_custom_output.json

# Sort with custom input/output
python sort.py --resume resumes/custom.txt --input custom_skills.json --output custom_sorted.json

# Match with custom input/output
python taskmatch.py --resume resumes/custom.txt --skills-file custom_skills.json --output-file custom_matches.json
```

## Benefits

- **No overwriting**: Each resume gets its own directory
- **Clean git repo**: `data/` and `resumes/` are ignored by git
- **Easy testing**: Test multiple resumes without file conflicts
- **Auto-creation**: Directories are created automatically as needed
- **Backward compatible**: Can still specify custom paths when needed
