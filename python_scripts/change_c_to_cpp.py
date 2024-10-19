import os

def add_extern_c_to_cpp_file(file_path):
    """Add extern "C" guards to the .cpp file content."""
    with open(file_path, 'r', encoding='utf-8') as file:
        content = file.read()

    # Add extern "C" guard if not already present
    if '#ifdef __cplusplus' not in content:
        extern_c_guard = (
            "#ifdef __cplusplus\n"
            "extern \"C\" {\n"
            "#endif\n\n"
        )
        extern_c_end = (
            "\n#ifdef __cplusplus\n"
            "}\n"
            "#endif\n"
        )
        
        # Insert the extern C guard at the start and end of the file
        content = extern_c_guard + content + extern_c_end
        
        with open(file_path, 'w', encoding='utf-8') as file:
            file.write(content)

def rename_and_modify_c_to_cpp(directory):
    """Rename all .c files to .cpp and add extern "C" block."""
    for root, dirs, files in os.walk(directory):  # Recursively go through directories
        for file in files:
            if file.endswith(".c"):  # Only process .c files
                c_file = os.path.join(root, file)  # Full path to the .c file
                cpp_file = os.path.join(root, file[:-2] + ".cpp")  # New .cpp file path
                
                # Rename the file
                os.rename(c_file, cpp_file)
                print(f"Renamed: {c_file} -> {cpp_file}")
                
                # Add extern "C" to the new .cpp file
                add_extern_c_to_cpp_file(cpp_file)
                print(f"Added extern \"C\" guard to: {cpp_file}")

# Set the directory you want to start from
directory = "C:/Users/augus/Desktop/MainFolder/University/Master/MscThesis/Project/Git/AI_Lora_Mobility/tflite-micro-arduino-examples/src/third_party/cmsis_nn/Source"  # Replace with your desired path
rename_and_modify_c_to_cpp(directory)
