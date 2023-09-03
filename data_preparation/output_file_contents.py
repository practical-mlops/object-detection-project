from kfp.components import InputPath


def output_file_contents(file_path: InputPath(str)):
    print(f"Printing contents of :{file_path}")

    with open(file_path, 'r') as f:
        print(f.read())
