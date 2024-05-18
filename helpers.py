def remove_duplicate_words(file_path):
    with open(file_path, 'r') as file:
        lines = file.readlines()

    unique_lines = []
    for line in lines:
        words = line.split()
        unique_words = list(set(words))
        unique_line = ' '.join(unique_words)
        unique_lines.append(unique_line + '\n')

    with open(file_path, 'w') as file:
        file.writelines(unique_lines)
