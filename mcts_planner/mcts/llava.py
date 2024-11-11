import torch
from llava.videollava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN
from llava.videollava.conversation import conv_templates, SeparatorStyle
from llava.videollava.model.builder import load_pretrained_model
from llava.videollava.utils import disable_torch_init
from llava.videollava.mm_utils import tokenizer_image_token, get_model_name_from_path, KeywordsStoppingCriteria

disable_torch_init()
llava_model_path = 'LanguageBind/Video-LLaVA-7B'
cache_dir = 'cache_dir'
llava_device = 'cuda'
load_4bit, load_8bit = True, False
model_name = get_model_name_from_path(llava_model_path)
llava_tokenizer, llava_model, llava_processor, _ = load_pretrained_model(llava_model_path, None, model_name, load_8bit, load_4bit, device=llava_device, cache_dir=cache_dir)
llava_video_processor = llava_processor['video']
conv_mode = "llava_v1"
llava_conv = conv_templates[conv_mode].copy()
roles = llava_conv.roles

def video_to_text(video, inp) -> str:
    video_tensor = llava_video_processor(video, return_tensors='pt')['pixel_values']
    if type(video_tensor) is list:
        tensor = [video.to(llava_model.device, dtype=torch.float16) for video in video_tensor]
    else:
        tensor = video_tensor.to(llava_model.device, dtype=torch.float16)

    print(f"{roles[1]}: {inp}")
    inp = ' '.join([DEFAULT_IMAGE_TOKEN] * llava_model.get_video_tower().config.num_frames) + '\n' + inp
    llava_conv.append_message(llava_conv.roles[0], inp)
    llava_conv.append_message(llava_conv.roles[1], None)
    prompt = llava_conv.get_prompt()
    input_ids = tokenizer_image_token(prompt, llava_tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).cuda()
    stop_str = llava_conv.sep if llava_conv.sep_style != SeparatorStyle.TWO else llava_conv.sep2
    keywords = [stop_str]
    stopping_criteria = KeywordsStoppingCriteria(keywords, llava_tokenizer, input_ids)

    with torch.inference_mode():
        output_ids = llava_model.generate(
            input_ids,
            images=tensor,
            do_sample=True,
            temperature=0.1,
            max_new_tokens=1024,
            use_cache=True,
            stopping_criteria=[stopping_criteria])

    outputs = llava_tokenizer.decode(output_ids[0, input_ids.shape[1]:]).strip()
    return outputs

# EXAMPLE
# video = 'videollava/serve/examples/sample_demo_1.mp4'
# inp = 'Did the robot complete the task of getting milk from the fridge? Answer with a score of 1-100.'
# text = video_to_text(video, inp)
# print(text)