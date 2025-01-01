from kfp import dsl
from kfp.dsl import Output, Input, Artifact


@dsl.component(
    base_image="python:3.11",
    packages_to_install=["pyjokes"]
)
def generate_joke(
        num_of_jokes: int,
        output_jokes: Output[Artifact]  # A
):
    import pyjokes
    import os

    # IMPORTANT: Make sure the directory is created first!
    os.makedirs(output_jokes.path, exist_ok=True)
    # Write jokes to the artifact path
    jokes_path = os.path.join(output_jokes.path, "jokes.txt")  # B
    with open(jokes_path, 'w') as f:
        for _ in range(num_of_jokes):
            joke = pyjokes.get_joke()
            f.write(joke + '\n')


@dsl.component
def count_words(
        jokes_file: Input[Artifact]  # C
) -> int:
    import os

    # Read from the artifact path
    jokes_path = os.path.join(jokes_file.path, "jokes.txt")  # D
    with open(jokes_path, 'r') as f:
        content = f.read()
        return len(content.split())


@dsl.component
def output_result(count: int) -> str:
    return f"Word count: {count}"


@dsl.pipeline(
    name="joke-pipeline",
    description="Pipeline that generates jokes and counts words"
)
def joke_pipeline(num_jokes: int = 3):
    jokes = generate_joke(num_of_jokes=num_jokes)
    count = count_words(jokes_file=jokes.outputs["output_jokes"])  # E
    output = output_result(count=count.output)


if __name__ == '__main__':
    from kfp import compiler

    compiler.Compiler().compile(
        pipeline_func=joke_pipeline,
        package_path='generate_jokes_pipeline.yaml'
    )
