from transformers import pipeline  # , BitsAndBytesConfig
import argparse
import sys
import logging


def parse_arguments():
    parser = argparse.ArgumentParser(
        description="This script paraphrases a given key field in a jsonl file."
    )
    parser.add_argument(
        "file",
        nargs="?",
        type=argparse.FileType("r"),
        default=sys.stdin,
        help="Input .csv or jsonl file or stdin",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="sbintuitions/tiny-lm-chat",  # BramVanroy/GEITje-7B-ultra
        help="LLM to use; default GEITJE ULTRA 7B.",
    )
    parser.add_argument(
        "--n",
        type=int,
        default=2,
        help="Number of paraphrases to generate; default 1.",
    )
    parser.add_argument(
        "--temp",
        type=float,
        default=0.8,
        help="Temperature (higher leads to more creative writing); default 0.8.",
    )
    parser.add_argument(
        "--top_k",
        type=int,
        default=10,
        help="Top k value; default 10.",
    )
    parser.add_argument(
        "--top_p",
        type=float,
        default=1.0,
        help="Top_p value default 1.0.",
    )
    parser.add_argument("--seed", type=int, default=None, help="Random seed.")
    args = parser.parse_args()
    return args


def paraphrase_cli():
    args = parse_arguments()
    logging.basicConfig(level=logging.INFO)
    logging.info(f"Using model: {args.model}")
    logging.info(f"Generating {args.n} paraphrases")
    logging.info(
        f"Using hyperparameters - Temp: {args.temp}, Top_k: {args.top_k}, Top_p: {args.top_p}"
    )
    logging.info(f"Input file: {args.file}")
    if args.seed is None:
        logging.info("Seed not implemented yet.")

    # See https://huggingface.co/docs/transformers/main_classes/text_generation for all hyperparameters
    pipe = pipeline(
        task="text-generation",
        model=args.model,  # does not work with all models because of chat_templates not set etc.
        # model_kwargs={"quantization_config" : BitsAndBytesConfig(load_in_4bit=True)},
        device_map="auto",
        max_new_tokens=500,
        do_sample=True,  # greedy decoding or not
        temperature=args.temp,  # modulates the next tokens probabilities
        num_return_sequences=args.n,
        top_k=args.top_k,
        top_p=args.top_p,
    )
    input_reader = prompt_formatting(args.file)  # generator of messages_lists
    for input in input_reader:
        raw_input = input[1]['content'].split("'")[1].strip()
        response = pipe(input)
        list_of_responses = [
            text["generated_text"][2]["content"].strip() for text in response
        ]
        for res in list_of_responses:
            intersection = set(res.split(' ')) & set(raw_input.split(' '))
            union = set(res.split(' ')) | set(raw_input.split(' '))
            iou = round(len(intersection) / len(union), 2)
            logging.info(f"Intersection over Union between original and generated text: {iou}")
            print(res.replace("\n", ""))
        print()


def prompt_formatting(texts):
    for text in texts:
        prompt = [
            {
                "role": "system",
                "content": "Je bent een schrijver die zinnen op andere manieren kan formuleren.",
            },
            {
                "role": "user",
                "content": f"Kan je de volgende zin anders schrijven: '{text}'",
            },
        ]
        yield prompt


if __name__ == "__main__":
    paraphrase_cli()
