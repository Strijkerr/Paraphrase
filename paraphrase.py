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
        help="One line per text to-be-paraphrased.",
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
    if args.seed:
        logging.warning("Seed not implemented yet.")

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
    
    for response in pipe(iter_prompts(args.file)):
        # each 'response' is a chain of messages (inc. the system message and prompt)
        for message in response:

            input_text = message["generated_text"][1]['content'].split("\n", maxsplit=1)[-1]
            response_text = message["generated_text"][2]["content"]
            
            logging.info(f"Similarity of paraphrase to original (IOU): {round(iou(input_text, response_text), 2)}")
            
            print(response.replace("\n", "\\n"))
        print()


def iou(text1, text2):
    set1 = set(text1.strip().lower().split())
    set2 = set(text2.strip().lower().split())
    return len(set1 & set2) / len(set1 | set2)

def iter_prompts(texts):
    for text in texts:
        prompt = [
            {
                "role": "system",
                "content": "Je bent een behulpzame assistent.",
            },
            {
                "role": "user",
                "content": f"Formuleer de onderstaande zin lichtjes, met behoud van betekenis (en verander niet te veel!): \n{text}",
            },
        ]
        yield prompt


if __name__ == "__main__":
    paraphrase_cli()
