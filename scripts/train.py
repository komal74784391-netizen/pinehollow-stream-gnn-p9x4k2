import argparse, yaml, torch
from src.model import StreamPrior

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True)
    args = parser.parse_args()
    cfg = yaml.safe_load(open(args.config))
    model = StreamPrior(hidden=cfg['hidden'], layers=cfg['layers'])
    opt = torch.optim.AdamW(model.parameters(), lr=cfg['lr'])
    for step in range(cfg['steps']):
        loss = torch.rand(1, requires_grad=True)
        opt.zero_grad(); loss.backward(); opt.step()
    print('trained', cfg['steps'], 'steps')

if __name__ == '__main__':
    main()
