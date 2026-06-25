import argparse
from pathlib import Path

import torch
from torch import nn

from models import ConvModel, MLPModel
from utils import (
    evaluate,
    get_device,
    make_dataloaders,
    plot_metrics,
    train,
    train_early_stopping,
)


def parse_args():
    parser = argparse.ArgumentParser(description='Chest X-ray classification training script')
    parser.add_argument('--data-dir', type=str, default='chest_xray_64', help='Path to the dataset root directory')
    parser.add_argument('--model', type=str, choices=['mlp', 'cnn'], default='cnn', help='Model architecture')
    parser.add_argument('--strategy', type=str, choices=['baseline', 'early_stopping', 'weight_decay'], default='baseline', help='Training strategy')
    parser.add_argument('--batch-size', type=int, default=32, help='Batch size for training and validation')
    parser.add_argument('--epochs', type=int, default=30, help='Number of training epochs')
    parser.add_argument('--lr', type=float, default=1e-3, help='Learning rate')
    parser.add_argument('--weight-decay', type=float, default=0.0, help='L2 weight decay for optimizer')
    parser.add_argument('--output-dir', type=str, default='output', help='Directory to save results and model weights')
    parser.add_argument('--no-cuda', action='store_true', help='Force CPU even when CUDA is available')
    parser.add_argument('--plot', action='store_true', help='Save training plots to output directory')
    return parser.parse_args()


def main():
    args = parse_args()
    device = get_device(force_cpu=args.no_cuda)

    print(f'Device: {device}')
    train_loader, val_loader = make_dataloaders(
        args.data_dir,
        batch_size=args.batch_size,
        num_workers=0,
    )

    if args.model == 'mlp':
        model = MLPModel()
    else:
        model = ConvModel()

    model = model.to(device)
    loss_fn = nn.CrossEntropyLoss()
    weight_decay = args.weight_decay if args.strategy == 'weight_decay' else 0.0
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=weight_decay)

    if args.strategy == 'early_stopping':
        train_losses, val_losses, train_acc, val_acc = train_early_stopping(
            model,
            train_loader,
            val_loader,
            optimizer,
            loss_fn,
            device,
            args.epochs,
            patience=5,
        )
    else:
        train_losses, val_losses, train_acc, val_acc = train(
            model,
            train_loader,
            val_loader,
            optimizer,
            loss_fn,
            device,
            args.epochs,
        )

    val_loss, val_accuracy = evaluate(model, val_loader, loss_fn, device)
    print(f'Final validation loss: {val_loss:.4f}, final validation accuracy: {val_accuracy:.2f}%')

    output_path = Path(args.output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    model_path = output_path / f'{args.model}_{args.strategy}.pth'
    torch.save(model.state_dict(), model_path)
    print(f'Saved trained model to: {model_path}')

    if args.plot:
        plot_metrics(train_losses, val_losses, train_acc, val_acc, output_dir=str(output_path))

    metrics_path = output_path / 'metrics.txt'
    with metrics_path.open('w', encoding='utf-8') as handle:
        handle.write(f'validation_loss: {val_loss:.4f}\n')
        handle.write(f'validation_accuracy: {val_accuracy:.2f}\n')
        handle.write(f'model: {args.model}\n')
        handle.write(f'strategy: {args.strategy}\n')
        handle.write(f'epochs: {len(train_losses)}\n')

    print(f'Saved metrics to: {metrics_path}')


if __name__ == '__main__':
    main()
