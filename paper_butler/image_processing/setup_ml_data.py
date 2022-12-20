import json
import os.path
from random import random
from typing import Any, List
import json
from pathlib import Path
import torch
from PIL import Image
from donut import DonutModel
from pdf2image import convert_from_path
import re
from transformers import DonutProcessor, VisionEncoderDecoderModel
from datasets import load_dataset
import torch
from transformers import VisionEncoderDecoderModel
from huggingface_hub import HfFolder
from transformers import Seq2SeqTrainingArguments, Seq2SeqTrainer
from basic.models import Document
from django.conf import settings
new_special_tokens = []
processor = None
model = None
model_name = "donut_test_paper_butler"


def run_prediction(sample, model, processor):
    # prepare inputs
    pixel_values = torch.tensor(sample["pixel_values"]).unsqueeze(0)
    task_prompt = "<s_rvlcdip>"
    decoder_input_ids = processor.tokenizer(task_prompt, add_special_tokens=False, return_tensors="pt").input_ids
    device = "cpu"
    # run inference
    outputs = model.generate(
        pixel_values.to(device),
        decoder_input_ids=decoder_input_ids.to(device),
        max_length=model.decoder.config.max_position_embeddings,
        early_stopping=True,
        pad_token_id=processor.tokenizer.pad_token_id,
        eos_token_id=processor.tokenizer.eos_token_id,
        use_cache=True,
        num_beams=1,
        bad_words_ids=[[processor.tokenizer.unk_token_id]],
        return_dict_in_generate=True,
    )

    # process output
    prediction = processor.batch_decode(outputs.sequences)[0]
    prediction = processor.token2json(prediction)

    # load reference target
    target = processor.token2json(sample["target_sequence"])
    return prediction, target


def transform_and_tokenize(sample, split="train", max_length=512, ignore_id=-100):
    # create tensor from image
    global processor
    try:
        pixel_values = processor(
            sample["image"], random_padding=split == "train", return_tensors="pt"
        ).pixel_values.squeeze()
    except Exception as e:
        print(sample)
        print(f"Error: {e}")
        return {}

    # tokenize document
    input_ids = processor.tokenizer(
        sample["text"],
        add_special_tokens=False,
        max_length=max_length,
        padding="max_length",
        truncation=True,
        return_tensors="pt",
    )["input_ids"].squeeze(0)

    labels = input_ids.clone()
    labels[labels == processor.tokenizer.pad_token_id] = ignore_id  # model doesn't need to predict pad token
    return {"pixel_values": pixel_values, "labels": labels, "target_sequence": sample["text"]}


def json2token(obj, new_special_tokens, update_special_tokens_for_json_key: bool = True, sort_json_key: bool = True):
    """
    Convert an ordered JSON object into a token sequence
    """
    if type(obj) == dict:
        if len(obj) == 1 and "text_sequence" in obj:
            return obj["text_sequence"]
        else:
            output = ""
            if sort_json_key:
                keys = sorted(obj.keys(), reverse=True)
            else:
                keys = obj.keys()
            for k in keys:
                if update_special_tokens_for_json_key:
                    new_special_tokens.append(fr"<s_{k}>") if fr"<s_{k}>" not in new_special_tokens else None
                    new_special_tokens.append(fr"</s_{k}>") if fr"</s_{k}>" not in new_special_tokens else None
                output += (
                        fr"<s_{k}>"
                        + json2token(obj[k], new_special_tokens, update_special_tokens_for_json_key, sort_json_key)
                        + fr"</s_{k}>"
                )
            return output
    elif type(obj) == list:
        return r"<sep/>".join(
            [json2token(item, new_special_tokens, update_special_tokens_for_json_key, sort_json_key) for item in obj]
        )
    else:
        # excluded special tokens for now
        obj = str(obj)
        if f"<{obj}/>" in new_special_tokens:
            obj = f"<{obj}/>"  # for categorical special tokens
        return obj


def preprocess_documents_for_donut(sample, new_special_tokens):
    # create Donut-style input
    text = json.loads(sample["text"])
    d_doc = "<s>" + json2token(text, new_special_tokens) + "</s>"
    # convert all images to RGB
    image = sample["image"].convert('RGB')
    return {"image": image, "text": d_doc}


def initialize_model_and_preprocessor():
    global model, processor
    if os.path.exists("test_model"):
        processor = DonutProcessor.from_pretrained("test_model")
        model = VisionEncoderDecoderModel.from_pretrained("test_model")
    else:
        processor = DonutProcessor.from_pretrained("naver-clova-ix/donut-base")
        model = VisionEncoderDecoderModel.from_pretrained("naver-clova-ix/donut-base")


def prepare_image(path):
    page = convert_from_path(path)[0]
    page_data = page.fp.read()
    size = page.size
    img = Image.frombytes("RGB", size, page_data)
    new_name = Path(Path(path).name).with_suffix(".jpg")
    out_path = Path(settings.MEDIA_ROOT).joinpath("ml").joinpath(new_name)
    img.save(out_path)
    return str(new_name)

def prepare_dataset():
    bad_keys = ["upload_date_time", "human_readable_id","id", "payed","file_obj"]
    new_data = []
    for d in Document.objects.exclude(used_for_training=True).all():
        data_dict = {}
        for k,v in d.__dict__.items():
            if not k.startswith("_") and not k in bad_keys:
                data_dict[k] = str(v)
        d.used_for_training = True
        d.save()
        img_path = prepare_image(d.file_obj.path)
        new_data.append({"file_name": img_path, "text": json.dumps(data_dict)})
    if len(new_data) == 0:
        return []
    else:
        with open(Path(settings.MEDIA_ROOT).joinpath("ml").joinpath("metadata.jsonl"), "w+") as out_:
            for entry in new_data:
                json.dump(entry, out_)
                out_.write('\n')
        dataset = load_dataset("imagefolder", data_dir=str(Path(settings.MEDIA_ROOT).joinpath("ml")), split="train")
        return dataset


def preprocess_dataset(dataset):
    new_special_tokens = []
    proc_dataset = dataset.map(preprocess_documents_for_donut, fn_kwargs={"new_special_tokens": new_special_tokens})
    processor.tokenizer.add_special_tokens({"additional_special_tokens": new_special_tokens + ["<s>"] + ["</s>"]})

    # we update some settings which differ from pretraining; namely the size of the images + no rotation required
    # resizing the image to smaller sizes from [1920, 2560] to [960,1280]
    processor.current_processor.size = [720, 960]  # should be (width, height)
    processor.current_processor.do_align_long_axis = False
    processed_dataset = proc_dataset.map(transform_and_tokenize, remove_columns=["image", "text"])
    processed_dataset = processed_dataset.train_test_split(test_size=0.1)
    return processed_dataset


def prepare_encoder_decoder(processed_dataset):
    new_emb = model.decoder.resize_token_embeddings(len(processor.tokenizer))
    print(f"New embedding size: {new_emb}")
    # Adjust our image size and output sequence lengths
    model.config.encoder.image_size = processor.current_processor.size[::-1]  # (height, width)
    model.config.decoder.max_length = len(max(processed_dataset["train"]["labels"], key=len))

    # Add task token for decoder to start
    model.config.pad_token_id = processor.tokenizer.pad_token_id
    model.config.decoder_start_token_id = processor.tokenizer.convert_tokens_to_ids([''])[0]

    # hyperparameters used for multiple args
    name = "donut-base-test"

    # Arguments for training
    training_args = Seq2SeqTrainingArguments(
        output_dir=name,
        num_train_epochs=3,
        learning_rate=2e-5,
        per_device_train_batch_size=2,
        weight_decay=0.01,
        fp16=False,
        logging_steps=100,
        save_total_limit=2,
        evaluation_strategy="no",
        save_strategy="epoch",
        predict_with_generate=True
    )

    # Create Trainer
    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=processed_dataset["train"],
    )
    return trainer


def train_model(trainer, model_name):
    trainer.train()
    processor.save_pretrained(model_name)
    model.save_pretrained(model_name)


def evaluate(processed_dataset):
    test_sample = processed_dataset["test"][0]
    prediction, target = run_prediction(test_sample,model,processor)
    print(f"Reference:\n {target}")
    print(f"Prediction:\n {prediction}")
    true_counter = 0
    total_counter = 0
    for sample in processed_dataset["test"]:
        prediction, target = run_prediction(sample,model,processor)
        for s in zip(prediction.values(), target.values()):
            if s[0] == s[1]:
                true_counter += 1
            total_counter += 1
    print(f"Accuracy: {(true_counter / total_counter) * 100}%")

def reset_all_documents():
    for d in Document.objects.all():
        d.used_for_training = False
        d.save()

def run_training_cylce():
    reset_all_documents()
    initialize_model_and_preprocessor()
    dataset = prepare_dataset()
    if len(dataset) > 0:
        preprocessed_dataset = preprocess_dataset(dataset)
        trainer = prepare_encoder_decoder(preprocessed_dataset)
        train_model(trainer, "test_model")
        evaluate(preprocessed_dataset)



def prepare():
    global new_special_tokens
    global processor
    from transformers import DonutProcessor

    import json
    from pathlib import Path
    # bad_keys = ["upload_date_time", "id", "human_readable_id", "payed"]
    # data = [{k: v for k, v in a.__dict__.items() if not k.startswith("_") and k not in bad_keys} for a in Document.objects.all()]
    # new_data=[]
    # for d in data:
    #    new_data.append({"file_name":Path(d["file_obj"]).name,"text":json.dumps(d)})
    # with open("/Users/holzmano/Documents/Projects/paper-butler/paper_butler/media/original/metadata.jsonl", "w+") as out_:
    #     for entry in new_data:
    #         json.dump(entry, out_)
    #         out_.write('\n')

    dataset = load_dataset("imagefolder", data_dir="/media/original/", split="all")
    new_special_tokens = []
    proc_dataset = dataset.map(preprocess_documents_for_donut, fn_kwargs={"new_special_tokens": new_special_tokens})

    processor = DonutProcessor.from_pretrained("naver-clova-ix/donut-base")

    # add new special tokens to tokenizer
    processor.tokenizer.add_special_tokens({"additional_special_tokens": new_special_tokens + ["<s>"] + ["</s>"]})

    # we update some settings which differ from pretraining; namely the size of the images + no rotation required
    # resizing the image to smaller sizes from [1920, 2560] to [960,1280]
    processor.current_processor.size = [720, 960]  # should be (width, height)
    processor.current_processor.do_align_long_axis = False
    processed_dataset = proc_dataset.map(transform_and_tokenize, remove_columns=["image", "text"])
    processed_dataset = processed_dataset.train_test_split(test_size=0.1)

    # Load model from huggingface.co
    model = VisionEncoderDecoderModel.from_pretrained("naver-clova-ix/donut-base")

    # Resize embedding layer to match vocabulary size
    new_emb = model.decoder.resize_token_embeddings(len(processor.tokenizer))
    print(f"New embedding size: {new_emb}")
    # Adjust our image size and output sequence lengths
    model.config.encoder.image_size = processor.current_processor.size[::-1]  # (height, width)
    model.config.decoder.max_length = len(max(processed_dataset["train"]["labels"], key=len))

    # Add task token for decoder to start
    model.config.pad_token_id = processor.tokenizer.pad_token_id
    model.config.decoder_start_token_id = processor.tokenizer.convert_tokens_to_ids([''])[0]

    # hyperparameters used for multiple args
    name = "donut-base-test"

    # Arguments for training
    training_args = Seq2SeqTrainingArguments(
        output_dir=name,
        num_train_epochs=3,
        learning_rate=2e-5,
        per_device_train_batch_size=2,
        weight_decay=0.01,
        fp16=False,
        logging_steps=100,
        save_total_limit=2,
        evaluation_strategy="no",
        save_strategy="epoch",
        predict_with_generate=True
    )

    # Create Trainer
    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=processed_dataset["train"],
    )

    trainer.train()
    processor.save_pretrained("test")
    test_sample = processed_dataset["test"][0]
    prediction, target = run_prediction(test_sample, model,processor)
    print(f"Reference:\n {target}")
    print(f"Prediction:\n {prediction}")
    true_counter = 0
    total_counter = 0
    for sample in processed_dataset["test"]:
        prediction, target = run_prediction(sample, model,processor)
        for s in zip(prediction.values(), target.values()):
            if s[0] == s[1]:
                true_counter += 1
            total_counter += 1
    print(f"Accuracy: {(true_counter / total_counter) * 100}%")


def one():
    processor = DonutProcessor.from_pretrained("naver-clova-ix/donut-base-finetuned-rvlcdip")
    model = VisionEncoderDecoderModel.from_pretrained("naver-clova-ix/donut-base-finetuned-rvlcdip")
    device = "cpu"
    model.to(device)
    # load document image
    image = load_pdf("/media/original/2022-11-21_20.17.23.pdf")
    # prepare decoder inputs
    task_prompt = "<s_rvlcdip>"
    decoder_input_ids = processor.tokenizer(task_prompt, add_special_tokens=False, return_tensors="pt").input_ids
    pixel_values = processor(image, return_tensors="pt").pixel_values
    outputs = model.generate(
        pixel_values.to(device),
        decoder_input_ids=decoder_input_ids.to(device),
        max_length=model.decoder.config.max_position_embeddings,
        early_stopping=True,
        pad_token_id=processor.tokenizer.pad_token_id,
        eos_token_id=processor.tokenizer.eos_token_id,
        use_cache=True,
        num_beams=1,
        bad_words_ids=[[processor.tokenizer.unk_token_id]],
        return_dict_in_generate=True,
    )

    sequence = processor.batch_decode(outputs.sequences)[0]
    sequence = sequence.replace(processor.tokenizer.eos_token, "").replace(processor.tokenizer.pad_token, "")
    sequence = re.sub(r"<.*?>", "", sequence, count=1).strip()  # remove first task start token
    print(processor.token2json(sequence))


def train():
    from transformers import AutoFeatureExtractor, DonutSwinModel
    import torch
    from datasets import load_dataset

    dataset = load_dataset("huggingface/cats-image")
    image = dataset["test"]["image"][0]

    feature_extractor = AutoFeatureExtractor.from_pretrained("https://huggingface.co/naver-clova-ix/donut-base")
    model = DonutSwinModel.from_pretrained("https://huggingface.co/naver-clova-ix/donut-base")

    inputs = feature_extractor(image, return_tensors="pt")

    with torch.no_grad():
        outputs = model(**inputs)

    last_hidden_states = outputs.last_hidden_state
    list(last_hidden_states.shape)


def two():
    processor = DonutProcessor.from_pretrained("naver-clova-ix/donut-base-finetuned-cord-v2")
    model = VisionEncoderDecoderModel.from_pretrained("naver-clova-ix/donut-base-finetuned-cord-v2")

    device = "cpu"
    model.to(device)
    # load document image
    image = load_pdf("test.pdf")

    # prepare decoder inputs
    task_prompt = "<s_cord-v2>"
    decoder_input_ids = processor.tokenizer(task_prompt, add_special_tokens=False, return_tensors="pt").input_ids

    pixel_values = processor(image, return_tensors="pt").pixel_values

    outputs = model.generate(
        pixel_values.to(device),
        decoder_input_ids=decoder_input_ids.to(device),
        max_length=model.decoder.config.max_position_embeddings,
        early_stopping=True,
        pad_token_id=processor.tokenizer.pad_token_id,
        eos_token_id=processor.tokenizer.eos_token_id,
        use_cache=True,
        num_beams=1,
        bad_words_ids=[[processor.tokenizer.unk_token_id]],
        return_dict_in_generate=True,
    )

    sequence = processor.batch_decode(outputs.sequences)[0]
    sequence = sequence.replace(processor.tokenizer.eos_token, "").replace(processor.tokenizer.pad_token, "")
    sequence = re.sub(r"<.*?>", "", sequence, count=1).strip()  # remove first task start token
    print(processor.token2json(sequence))


def create_setup():
    pass


def create_groundtruth_data():
    pass


def init():
    pretrained_model = DonutModel.from_pretrained("naver-clova-ix/donut-base-finetuned-rvlcdip")
    pretrained_model.eval()
    return pretrained_model


def init_2():
    pretrained_model = DonutModel.from_pretrained("naver-clova-ix/donut-base-finetuned-cord-v2")
    pretrained_model.eval()
    return pretrained_model


def demo_process(model, input_img):
    output = model.inference(image=input_img, prompt="<s_rvlcdip>")
    return output["predictions"][0]


def load_pdf(path):
    path = Path(settings.MEDIA_ROOT).joinpath(path)
    page = convert_from_path(path)[0]
    page_data = page.fp.read()
    size = page.size
    img = Image.frombytes("RGB", size, page_data)
    return img


def scale_image(img, downscale):
    return img.resize(tuple([int(a / downscale) for a in img.size]), Image.Resampling.LANCZOS)


def output_to_file(output, name):
    with open(name, "w+") as out_:
        json.dump(output, out_)


# two()
# one()
# model = init_2()
import time


def test_run():
    classification_model = DonutModel.from_pretrained("naver-clova-ix/donut-base-finetuned-rvlcdip")
    classification_model.eval()
    parser_model = DonutModel.from_pretrained("naver-clova-ix/donut-base-finetuned-cord-v2")
    parser_model.eval()
    img = load_pdf("test.pdf")
    for i in range(1, 100):
        img_new = scale_image(img, i)
        img_new.save(f"test_data/test_{img_new.size[0]}-{img_new.size[1]}.jpg")
        start_class = time.time()
        output = classification_model.inference(image=img_new, prompt="<s_rvlcdip>")["predictions"][0]
        output_to_file(output, f"test_data/test_class_{img_new.size[0]}-{img_new.size[1]}.json")
        timeit_class = int(time.time() - start_class)
        print(f"Classification had {timeit_class} seconds for dimensions {img_new.size[0]}x{img_new.size[1]} pixels")
        start_class = time.time()
        output = parser_model.inference(image=img_new, prompt="<s_cord-v2>")["predictions"][0]
        output_to_file(output, f"test_data/test_parse_{img_new.size[0]}-{img_new.size[1]}.json")
        timeit_class = int(time.time() - start_class)
        print(f"Parsing had {timeit_class} seconds for dimensions {img_new.size[0]}x{img_new.size[1]} pixels")

# test_run()
