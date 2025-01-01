import kfp.dsl as dsl


@dsl.component(
    packages_to_install=["pyjokes"],
    base_image="python:3.11"
)
def generate_joke() -> str:
    import pyjokes
    return pyjokes.get_joke()


@dsl.component
def count_words(input: str) -> int:
    return len(input.split())


@dsl.component
def output_result(count: int) -> str:
    return f"Word count: {count}"


@dsl.pipeline(
    name="joke_pipeline",
    description="simple pipeline demonstrating data passing")  # D
def pipeline():
    generate_joke_op = generate_joke()
    count_word_op = count_words(input=generate_joke_op.output)  # E
    output_result_op = output_result(count=count_word_op.output)  # E


if __name__ == '__main__':
    from kfp import compiler

    compiler.Compiler().compile(
        pipeline_func=pipeline,
        package_path='_pipeline.yaml'
    )
