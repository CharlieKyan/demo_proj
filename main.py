from train import traingpt_open_ended
import torch
from eval import eval_gpt_open_ended
from model import VQAModelWithKG
from dataloader import SlakeDataset
from torch.utils.data import DataLoader
import argparse
import numpy as np
import random
import os

def set_random_seeds(random_seed=0):

    torch.manual_seed(random_seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(random_seed)
    random.seed(random_seed)

def parse_argument():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_type", type=str, default="gpt2-xl", choices=("gpt2-xl", "microsoft/biogpt","stanford-crfm/BioMedLM"))
    parser.add_argument("--setting", type=str, default="frozen", choices=("lora", "frozen",'prefix_tuning',"p_tuning","prompt_tuning", "IA3","unfrozen"))
    parser.add_argument("--ablation", type=str, default="none", choices=("remove_question", "remove_visual",'replace_visual',"swap", "remove_kg", "kg_only"))
    parser.add_argument("--clip_mapping", type=str, default="mlp", choices=("transformer", "mlp","none"))
    parser.add_argument("--prefix_length", type=int, default=8)
    parser.add_argument(
        "--dataset_path", type=str, default="datasets/"
    )
    parser.add_argument("--kg_path", type=str, default="datasets/slake/KG/")
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--dataset", type=str, default='slake', choices=('pathvqa', 'slake'))
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--num_warmup_steps", type=int, default=600)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--iters_to_accumulate", type=int, default=4)
    parser.add_argument("--validation_step", type=int, default=1000)
    parser.add_argument("--out_dir", default="checkpoints")
    parser.add_argument("--checkpoint", type=str)
    parser.add_argument("--eval", dest="eval", action="store_true")

    parser.add_argument("--verbose", dest="verbose", action="store_true")
    parser.add_argument("--data_partition", type=str, default="slake", choices=("pathvqa", "slake"))
    parser.add_argument("--num_layers", type=int, default=3)
    parser.add_argument("--prefix_size", type=int, default=512)
    parser.add_argument("--kg_size", type=int, default=12288)
    parser.add_argument("--patience", type=int, default=5)

    args = parser.parse_args()

    set_random_seeds(args.seed)
    return args

if __name__ == "__main__":
    os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
    args = parse_argument()
    # Create the output directory if it does not exist
    suffix = f"v5_{args.data_partition}_prefixlength_{args.prefix_length}_mapping_{args.clip_mapping}_seed_{args.seed}_gpttype_{args.model_type.replace('/','')}_setting_{args.setting}_dataset_{args.dataset}"
    args.out_dir = os.path.join('checkpoints', suffix)
    if not os.path.exists(args.out_dir):
        os.makedirs(args.out_dir)

    train_dataset = SlakeDataset(data_path=args.dataset_path+args.dataset+'/', kg_path=args.kg_path ,split="train",prefix_length=args.prefix_length,model_type=args.model_type)#,abl=args.ablation)
    val_dataset = SlakeDataset(data_path=args.dataset_path+args.dataset+'/', kg_path=args.kg_path, split="val",prefix_length=args.prefix_length,model_type=args.model_type)#, abl=args.ablation)
    test_dataset = SlakeDataset(data_path=args.dataset_path+args.dataset+'/', kg_path=args.kg_path, split="test",prefix_length=args.prefix_length,model_type=args.model_type,like_test=True)

    if args.ablation != "none":
        pass # Implement ablation here
    else:
        # prefix_len, clip_len, prefix_size, kg_size, kg_len, num_layers, setting, clip_mapping, args = None
        model = VQAModelWithKG(
            prefix_len=args.prefix_length,
            clip_len=4,
            kg_size=args.kg_size,
            kg_len=16,
            setting=args.setting,
            clip_mapping=args.clip_mapping,
            args=args,
            prefix_size= args.prefix_size,
            num_layers=args.num_layers,
        )

    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, drop_last=True)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, drop_last=True)

    if not args.eval:
        model = traingpt_open_ended(train_loader, val_loader, model, args)
    else:
        checkpoint = os.path.join(args.out_dir, f"open_ended_latest.pt")
        print(checkpoint)
        if args.verbose:
            print(f">> Loading pre-trained model {checkpoint}!")
        if os.path.exists(checkpoint):
            model.load_state_dict(
                torch.load(checkpoint, map_location=torch.device("cuda:5")), strict=False
            )
        else:
            raise ValueError("Please provide valid path for loading checkpoint")
        eval_gpt_open_ended(model, test_dataset,args)
        # checkpoints/v5_slake_prefixlength_8_mapping_mlp_seed_0_gpttype_gpt2-xl_setting_frozen_dataset_slake/open_ended_latest.pt
        # checkpoints/v5_slake_prefixlength_8_mapping_mlp_seed_0_gpttype_gpt2-xl_setting_lora_dataset_slake/open_ended_latest.pt