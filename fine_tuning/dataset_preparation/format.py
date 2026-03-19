def format_example(example):
    if example["input"] != "":
        text = f"""### Instruction:
        {example["instruction"]}

        ### Input:
        {example["input"]}

        ### Response:
        {example["output"]}"""
    else:
        text = f"""### Instruction:
        {example["instruction"]}

        ### Response:
        {example["output"]}"""
    return text
