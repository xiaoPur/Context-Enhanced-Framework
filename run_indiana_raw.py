import argparse
import json
import os
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
import torchvision.models as tv_models

from datasets import IUXRAY, IndianaRawIUXRAY, MIMIC
from evaluation import compute_report_metrics, write_report_outputs
from losses import CELossTotal
from models import CNN, MVCNN, TNN, Classifier, Context, Generator
from qwen_postprocess import rewrite_reports_with_qwen
from utils import data_to_device, load, save, train, test


def parse_args():
    parser = argparse.ArgumentParser(description="Train or evaluate the context-enhanced IU X-Ray model.")
    parser.add_argument("--phase", choices=["train", "infer"], default="train")
    parser.add_argument("--dataset-name", choices=["indiana_raw", "iuxray", "mimic"], default="indiana_raw")
    parser.add_argument("--data-root", required=True, help="Dataset root directory.")
    parser.add_argument("--output-dir", default="outputs", help="Directory for checkpoints and evaluation outputs.")
    parser.add_argument("--checkpoint-path", default="", help="Checkpoint path override.")
    parser.add_argument("--split-file", default="", help="Optional persistent split json path.")
    parser.add_argument("--reports-csv", default="indiana_reports.csv")
    parser.add_argument("--projections-csv", default="indiana_projections.csv")
    parser.add_argument("--images-dir", default="images_normalized")
    parser.add_argument("--label-file", default="iu_xray/file2label.json")
    parser.add_argument("--external-label-file", default="", help="Optional external label mapping file.")
    parser.add_argument("--nounphrase-file", default="tools/count_nounphrase.json")
    parser.add_argument("--vocab-file", default="iu_xray/nlmcxr_unigram_1000.model")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--eval-batch-size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--train-size", type=float, default=0.7)
    parser.add_argument("--val-size", type=float, default=0.1)
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--input-size", type=int, nargs=2, default=(256, 256))
    parser.add_argument("--max-views", type=int, default=2)
    parser.add_argument("--max-len", type=int, default=300)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--lr", type=float, default=6e-5)
    parser.add_argument("--weight-decay", type=float, default=1e-2)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--embed-dim", type=int, default=256)
    parser.add_argument("--fwd-dim", type=int, default=256)
    parser.add_argument("--num-heads", type=int, default=8)
    parser.add_argument("--text-layers", type=int, default=1)
    parser.add_argument("--gen-heads", type=int, default=1)
    parser.add_argument("--gen-layers", type=int, default=12)
    parser.add_argument("--milestones", type=int, nargs="*", default=[25])
    parser.add_argument("--threshold", type=float, default=0.15)
    parser.add_argument("--device", default="", help="cuda, cuda:0, cpu. Defaults to cuda when available.")
    parser.add_argument("--pretrained-backbone", action="store_true", help="Use torchvision pretrained DenseNet121 weights.")
    parser.add_argument("--reload", action="store_true", help="Resume optimizer and scheduler states from checkpoint.")
    parser.add_argument("--run-eval", action="store_true", help="Run standard generation metrics after training/inference.")
    parser.add_argument(
        "--include-paper-metrics",
        action="store_true",
        help="Also compute METEOR and CIDEr with optional server-side pycocoevalcap dependencies.",
    )
    parser.add_argument("--run-qwen-eval", action="store_true", help="Run Qwen post-processing evaluation after standard generation.")
    parser.add_argument("--qwen-model-path", default="", help="Local Qwen2.5-7B-Instruct directory.")
    parser.add_argument("--qwen-max-new-tokens", type=int, default=256)
    parser.add_argument("--qwen-temperature", type=float, default=0.0)
    return parser.parse_args()


def seed_everything(seed):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_device(device_arg):
    if device_arg:
        return torch.device(device_arg)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def build_dataset_triplet(args):
    input_size = tuple(args.input_size)
    common = {
        "input_size": input_size,
        "max_views": args.max_views,
        "max_len": args.max_len,
        "sources": ["image", "caption", "label", "history"],
        "targets": ["caption", "label"],
    }

    if args.dataset_name == "indiana_raw":
        split_file = args.split_file or os.path.join(args.output_dir, "indiana_raw_splits.json")
        train_data = IndianaRawIUXRAY(
            directory=args.data_root,
            reports_csv=args.reports_csv,
            projections_csv=args.projections_csv,
            images_dir=args.images_dir,
            random_transform=True,
            split="train",
            split_file=split_file,
            train_size=args.train_size,
            val_size=args.val_size,
            test_size=args.test_size,
            seed=args.seed,
            vocab_file=args.vocab_file,
            label_file=args.label_file,
            external_label_file=args.external_label_file or None,
            nounphrase_file=args.nounphrase_file,
            **common,
        )
        val_data = IndianaRawIUXRAY(
            directory=args.data_root,
            reports_csv=args.reports_csv,
            projections_csv=args.projections_csv,
            images_dir=args.images_dir,
            random_transform=False,
            split="val",
            split_file=split_file,
            train_size=args.train_size,
            val_size=args.val_size,
            test_size=args.test_size,
            seed=args.seed,
            vocab_file=args.vocab_file,
            label_file=args.label_file,
            external_label_file=args.external_label_file or None,
            nounphrase_file=args.nounphrase_file,
            **common,
        )
        test_data = IndianaRawIUXRAY(
            directory=args.data_root,
            reports_csv=args.reports_csv,
            projections_csv=args.projections_csv,
            images_dir=args.images_dir,
            random_transform=False,
            split="test",
            split_file=split_file,
            train_size=args.train_size,
            val_size=args.val_size,
            test_size=args.test_size,
            seed=args.seed,
            vocab_file=args.vocab_file,
            label_file=args.label_file,
            external_label_file=args.external_label_file or None,
            nounphrase_file=args.nounphrase_file,
            **common,
        )
    elif args.dataset_name == "iuxray":
        dataset = IUXRAY(
            args.data_root,
            input_size,
            True,
            max_views=args.max_views,
            sources=common["sources"],
            targets=common["targets"],
            max_len=args.max_len,
            vocab_file=os.path.basename(args.vocab_file),
        )
        train_data, val_data, test_data = dataset.get_subsets(seed=args.seed)
    else:
        dataset = MIMIC(
            args.data_root,
            input_size,
            True,
            max_views=args.max_views,
            sources=common["sources"],
            targets=common["targets"],
            max_len=args.max_len,
        )
        train_data, val_data, test_data = dataset.get_subsets(seed=args.seed, generate_splits=True)

    return train_data, val_data, test_data


def build_model(dataset, args):
    backbone = tv_models.densenet121(pretrained=args.pretrained_backbone)
    num_labels = len(dataset[0][1][1])

    cnn = CNN(backbone, "DenseNet121")
    cnn = MVCNN(cnn)
    tnn = TNN(
        embed_dim=args.embed_dim,
        num_heads=args.num_heads,
        fwd_dim=args.fwd_dim,
        dropout=args.dropout,
        num_layers=args.text_layers,
        num_tokens=len(dataset.vocab),
        num_posits=dataset.max_len,
    )
    cls_model = Classifier(
        num_topics=num_labels,
        num_states=2,
        cnn=cnn,
        tnn=tnn,
        fc_features=1024,
        embed_dim=args.embed_dim,
        num_heads=args.gen_heads,
        dropout=args.dropout,
    )
    gen_model = Generator(
        num_tokens=len(dataset.vocab),
        num_posits=dataset.max_len,
        embed_dim=args.embed_dim,
        num_heads=args.gen_heads,
        fwd_dim=args.fwd_dim,
        dropout=args.dropout,
        num_layers=args.gen_layers,
    )
    return Context(cls_model, gen_model, num_labels, args.embed_dim)


def build_comment(dataset, args):
    return f"MaxView{args.max_views}_NumLabel{len(dataset[0][1][1])}_History"


def decode_sequence(vocab, sequence, pad_id):
    tokens = []
    for token in sequence:
        token_id = int(token)
        if token_id in {pad_id, vocab.bos_id()}:
            continue
        if token_id == vocab.eos_id():
            break
        tokens.append(token_id)
    if not tokens:
        return ""
    return " ".join(vocab.id_to_piece(token_id) for token_id in tokens).replace("\u2581", " ").strip()


def build_dataloaders(train_data, val_data, test_data, args, device):
    pin_memory = device.type == "cuda"
    train_loader = data.DataLoader(
        train_data,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        drop_last=True,
        pin_memory=pin_memory,
    )
    val_loader = data.DataLoader(
        val_data,
        batch_size=args.eval_batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=pin_memory,
    )
    test_loader = data.DataLoader(
        test_data,
        batch_size=args.eval_batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=pin_memory,
    )
    return train_loader, val_loader, test_loader


def evaluate_generation(data_loader, dataset, model, device, threshold, include_paper_metrics=False):
    model.eval()
    records = []
    references = []
    hypotheses = []
    pad_id = dataset.vocab.pad_id()
    record_offset = 0

    with torch.no_grad():
        for source, target in data_loader:
            source = data_to_device(source, str(device))
            target = data_to_device(target, str(device))
            batch_size = target[0].shape[0]
            generated, label_scores = model(image=source[0], history=source[3], threshold=threshold, max_len=dataset.max_len)

            generated = generated.detach().cpu()
            captions = target[0].detach().cpu()
            label_scores = label_scores.detach().cpu()

            for row_idx in range(batch_size):
                dataset_record = dataset.records[record_offset + row_idx] if hasattr(dataset, "records") else {"uid": str(record_offset + row_idx), "history": ""}
                reference_text = decode_sequence(dataset.vocab, captions[row_idx], pad_id)
                hypothesis_text = decode_sequence(dataset.vocab, generated[row_idx], pad_id)
                references.append(reference_text)
                hypotheses.append(hypothesis_text)
                records.append(
                    {
                        "uid": dataset_record.get("uid", str(record_offset + row_idx)),
                        "history": dataset_record.get("history", ""),
                        "reference": reference_text,
                        "hypothesis": hypothesis_text,
                        "predicted_topics": label_scores[row_idx].tolist(),
                    }
                )
            record_offset += batch_size

    metrics = compute_report_metrics(references, hypotheses, include_paper_metrics=include_paper_metrics)
    return references, hypotheses, records, metrics


def maybe_run_qwen(records, output_dir, args):
    if not args.run_qwen_eval:
        return None
    if not args.qwen_model_path:
        raise ValueError("--run-qwen-eval requires --qwen-model-path")

    qwen_records = rewrite_reports_with_qwen(
        records=records,
        model_name_or_path=args.qwen_model_path,
        max_new_tokens=args.qwen_max_new_tokens,
        temperature=args.qwen_temperature,
    )
    references = [record["reference"] for record in qwen_records]
    hypotheses = [record["qwen_hypothesis_normalized"] for record in qwen_records]
    metrics = compute_report_metrics(references, hypotheses, include_paper_metrics=args.include_paper_metrics)
    write_report_outputs(output_dir, references, hypotheses, metrics, qwen_records, prefix="qwen")
    return metrics


def run_evaluation(test_loader, test_data, model, args, output_dir):
    references, hypotheses, records, metrics = evaluate_generation(
        test_loader,
        test_data,
        model,
        get_device(args.device),
        args.threshold,
        include_paper_metrics=args.include_paper_metrics,
    )
    write_report_outputs(output_dir, references, hypotheses, metrics, records)
    qwen_metrics = maybe_run_qwen(records, output_dir, args) if args.run_qwen_eval else None
    return metrics, qwen_metrics


def default_checkpoint_path(args, dataset):
    comment = build_comment(dataset, args)
    file_name = f"{args.dataset_name}_Context_DenseNet121_{comment}.pt"
    checkpoint_dir = Path(args.output_dir) / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    return str(checkpoint_dir / file_name)


def main():
    args = parse_args()
    seed_everything(args.seed)
    device = get_device(args.device)
    args.device = str(device)

    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    train_data, val_data, test_data = build_dataset_triplet(args)
    train_loader, val_loader, test_loader = build_dataloaders(train_data, val_data, test_data, args, device)

    model = build_model(train_data, args)
    if device.type == "cuda":
        model = model.to(device)
        if torch.cuda.device_count() > 1:
            model = nn.DataParallel(model)
    else:
        model = model.to(device)

    optimizer = optim.AdamW(filter(lambda param: param.requires_grad, model.parameters()), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=args.milestones)
    criterion = CELossTotal(ignore_index=train_data.vocab.pad_id())

    checkpoint_path = args.checkpoint_path or default_checkpoint_path(args, train_data)
    best_metric = float("inf")
    last_epoch = -1

    if args.reload or args.phase == "infer":
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"checkpoint not found: {checkpoint_path}")
        last_epoch, stats = load(checkpoint_path, model, optimizer if args.reload else None, scheduler if args.reload else None)
        if stats:
            best_metric = stats[0]

    if args.phase == "train":
        scaler = torch.cuda.amp.GradScaler(enabled=device.type == "cuda")
        for epoch in range(last_epoch + 1, args.epochs):
            print(f"Epoch: {epoch}")
            train_loss = train(
                train_loader,
                model,
                optimizer,
                criterion,
                device=args.device,
                kw_src=["image", "caption", "label", "history"],
                scaler=scaler,
            )
            val_loss = test(
                val_loader,
                model,
                criterion,
                device=args.device,
                kw_src=["image", "caption", "label", "history"],
                return_results=False,
            )
            test_loss = test(
                test_loader,
                model,
                criterion,
                device=args.device,
                kw_src=["image", "caption", "label", "history"],
                return_results=False,
            )
            scheduler.step()
            print(json.dumps({"train_loss": train_loss, "val_loss": val_loss, "test_loss": test_loss}, ensure_ascii=False))

            if val_loss < best_metric:
                best_metric = val_loss
                save(checkpoint_path, model, optimizer, scheduler, epoch, (val_loss, test_loss))
                print(f"Saved best checkpoint to: {checkpoint_path}")

        if os.path.exists(checkpoint_path):
            load(checkpoint_path, model)

    if args.phase == "infer" or args.run_eval or args.run_qwen_eval:
        metrics, qwen_metrics = run_evaluation(test_loader, test_data, model, args, args.output_dir)
        print(json.dumps({"metrics": metrics, "qwen_metrics": qwen_metrics}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
