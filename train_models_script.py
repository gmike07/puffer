from config_creator import get_config, create_config
from tqdm import tqdm
import argparse
import torch
from data_iterator import DataIterator
from models import SL_Model


def save_cpp_model(model, model_path, CONFIG):
    example = torch.rand(1, CONFIG['nn_input_size']).double()
    traced_script_module = torch.jit.trace(model.model, example, check_trace=False)
    traced_script_module.save(model_path)


def train_sl(model, loader):
    CONFIG = get_config()
    for epoch in range(CONFIG['epochs']):
        pbar = tqdm(iterable=iter(loader), ncols=200)
        for (chunks, metrics) in pbar:
            predictions = model(chunks)
            loss = model.loss_metrics(predictions, metrics)
            model.optimizer.zero_grad()
            loss.backward()
            model.optimizer.step()
            pbar.set_description_str('epoch #{}'.format(epoch))
            pbar.set_postfix(loss=loss.mean().item())
        pbar.close()
        filename = f"sl_weights_{str(epoch)}_abr_{CONFIG['abr']}_v{str(CONFIG['version'])}_{CONFIG['scoring_function_type']}.pt"
        torch.save({
            'model_state_dict': model.model.state_dict()
        }, f"{CONFIG['weights_path']}{filename}")
        save_cpp_model(model, f"{CONFIG['weights_cpp_path']}{filename}", CONFIG)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-ip', "--input_dir", default='../cc_monitoring/')
    parser.add_argument('-yid', "--yaml_input_dir", default='/home/mike/puffer/helper_scripts/')
    parser.add_argument("--abr", default='')
    args = parser.parse_args()
    create_config(args.input_dir, args.yaml_input_dir, args.abr)
    model = SL_Model()    
    iterator = DataIterator(remove_bad=False, output_type='ssim')
    print('training all files...')
    train_sl(model, iterator)
    iterator = DataIterator(remove_bad=True, output_type='ssim')
    print('training good files...')
    train_sl(model, iterator)


if __name__ == '__main__':
    main()
