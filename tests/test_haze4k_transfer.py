"""Synthetic software checks only; no real Haze4K performance claims."""
import copy
import hashlib
import json
from pathlib import Path
import shutil

import numpy as np
from PIL import Image
import pytest
import torch

torch.set_num_threads(1)

from src.haze4k_transfer import prepare as prep
from src.haze4k_transfer.compare import align, bootstrap, main as compare, load_result
from src.haze4k_transfer.dataset import HazeDataset, tensor
from src.haze4k_transfer.eval import main as evaluate
from src.haze4k_transfer.models import assert_initial_backbone_equivalence, build_model, canonical_model_config
from src.haze4k_transfer.precision import cap_scaler, cap_scale_state, validate_scale_policy
from src.haze4k_transfer.recover import main as recover, validate_recovery_sources, LEGACY_TRAIN_SHA256, LEGACY_PREPARE_SHA256
from src.haze4k_transfer.train import accumulation_sample_count, normalized_backward_loss, main as train, validate_config
from src.haze4k_transfer.utils import METRICS, ROOT, load_yaml, protocol, save_yaml, source_hashes, torch_load
from src.v16.losses import CharbonnierLoss
from src.v16.metrics import batch_metrics, per_image_psnr, per_image_ssim


def make_data(root, n=10, nt=3, size=(32, 32)):
    for official, count, offset in (("train", n, 0), ("test", nt, 100000)):
        for role in ("haze", "gt", "trans"):
            (root / official / role).mkdir(parents=True)
        for i in range(1, count + 1):
            rng = np.random.default_rng(i + offset)
            gt = rng.integers(0, 220, (*size, 3), dtype=np.uint8)
            haze = np.minimum(gt.astype(int) + 20, 255).astype(np.uint8)
            Image.fromarray(gt).save(root / official / f"gt/{i}.png")
            Image.fromarray(haze).save(root / official / f"haze/{i}_0.68_0.66.png")
        # A corrupt image in trans must never be read/scanned as a model source.
        (root / official / "trans/corrupt.png").write_bytes(b"unused trans")
    (root / "train/haze/.DS_Store").write_bytes(b"metadata")
    return root


def test_pairing_grouping_determinism_and_trans(tmp_path):
    root = make_data(tmp_path / "data")
    shutil.copyfile(root / "train/haze/1_0.68_0.66.png", root / "train/haze/1_0.77_0.9.png")
    (root / "train/gt/notes.txt").write_text("not an image")
    split = tmp_path / "split"
    m = prep.prepare(root, split, synthetic=True)
    before = {p.name: p.read_bytes() for p in split.iterdir()}
    prep.prepare(root, split, synthetic=True)
    assert before == {p.name: p.read_bytes() for p in split.iterdir()}
    split2 = tmp_path / "split2"
    m2 = prep.prepare(root, split2, synthetic=True)
    assert m["hashes"] == m2["hashes"]
    _, rows, inventory = prep.validate_bundle(split)
    assert not {r['clean_id'] for r in rows['train']} & {r['clean_id'] for r in rows['validation']}
    assert sum(len(rows[s]) for s in ('train', 'validation')) == 11
    assert len(rows['test']) == 3
    assert not any('trans' in p or '.DS_Store' in p for p in inventory)
    assert (root / 'train/haze/.DS_Store').read_bytes() == b'metadata'
    assert any(r['clean_id'] == '1' and r['gt_relative'] == 'test/gt/1.png' for r in rows['test'])
    assert not any(m['report']['cross_official_content_overlap'].values())
    assert any(p['path'].endswith('notes.txt') for p in m['report']['train']['non_images'])
    bad = copy.deepcopy(rows['train'])
    bad.append(bad[0])
    with pytest.raises(ValueError, match='Duplicate'):
        prep.check_rows(bad, 'train')
    bad = copy.deepcopy(rows['train'])
    bad[0]['gt_relative'] = 'train/gt/999.png'
    with pytest.raises(ValueError, match='Incorrect pairing'):
        prep.check_rows(bad, 'train')
    with pytest.raises(ValueError, match='refusing rewrite'):
        prep.prepare(root, split, seed=1, synthetic=True)
    (split / 'train.tsv').write_bytes((split / 'train.tsv').read_bytes() + b'\n')
    with pytest.raises(ValueError, match='hash mismatch'):
        prep.validate_bundle(split)


def test_current_scale_2700_300_1000(tmp_path):
    root = make_data(tmp_path / 'data', 3000, 1000, size=(2, 2))
    before = {p.name: prep.sha256(p) for p in (root / 'test/gt').iterdir()}
    m = prep.prepare(root, tmp_path / 'split', synthetic=True)
    assert m['counts'] == {'train': 2700, 'validation': 300, 'test': 1000}
    assert before == {p.name: prep.sha256(p) for p in (root / 'test/gt').iterdir()}


@pytest.mark.parametrize('kind,pattern', [('missing', 'Missing/ambiguous'), ('ambiguous', 'Ambiguous GT'),
                                         ('size', 'size mismatch'), ('corrupt', 'Cannot fully decode'), ('unused', 'Unused GT')])
def test_bad_data(tmp_path, kind, pattern):
    root = make_data(tmp_path / 'data')
    if kind == 'missing':
        (root / 'train/gt/1.png').unlink()
    elif kind == 'ambiguous':
        Image.new('RGB', (32, 32)).save(root / 'train/gt/1.jpg')
    elif kind == 'size':
        Image.new('RGB', (16, 17)).save(root / 'train/gt/1.png')
    elif kind == 'corrupt':
        p = root / 'train/haze/1_0.68_0.66.png'
        p.write_bytes(p.read_bytes()[:70])  # Valid-looking header, truncated pixels.
    elif kind == 'unused':
        Image.new('RGB', (32, 32)).save(root / 'train/gt/999.png')
    with pytest.raises(ValueError, match=pattern):
        prep.prepare(root, tmp_path / 'split', synthetic=True)


def test_alias_overlap_and_content_immutability(tmp_path):
    root = make_data(tmp_path / 'data')
    (root / 'train/hazy').mkdir()
    with pytest.raises(ValueError, match='exactly one'):
        prep.scan(root)
    (root / 'train/hazy').rmdir()
    shutil.copyfile(root / 'train/gt/1.png', root / 'test/gt/1.png')
    split = tmp_path / 'split'
    with pytest.raises(ValueError, match='requires user review'):
        prep.prepare(root, split, synthetic=True)
    assert not (split / 'split_metadata.json').exists()
    report = json.loads((split / 'overlap_review_required.json').read_text())
    assert report['cross_official_content_overlap']['sha256']
    prep.prepare(root, split, accept_content_overlap=True, synthetic=True)
    _, rows, inventory = prep.validate_bundle(split)
    Image.new('RGB', (32, 32)).save(root / 'train/gt/1.png')
    with pytest.raises(ValueError, match='changed since'):
        prep.verify_content(root, rows['train'] + rows['validation'], inventory)
    # Same RGB content in a differently encoded file is caught separately.
    Image.open(root / 'train/gt/2.png').save(root / 'test/gt/2.png', compress_level=0)
    _, _, report = prep.scan(root)
    assert report['cross_official_content_overlap']['rgb_sha256']


def test_transforms_and_pil_protocol(tmp_path):
    root = make_data(tmp_path / 'data', n=3, nt=2, size=(9, 7))
    rows, _, _ = prep.scan(root)
    train = HazeDataset(root, rows['train'], 'train', patch_size=32)
    sample = train[0]
    assert sample['input'].shape == sample['target'].shape == (3, 32, 32)
    # Haze = GT + 20, preserved by every synchronized crop/pad/flip/rotation.
    assert torch.allclose(sample['input'] - sample['target'], torch.full((3, 32, 32), 20/255), atol=1e-7)
    dataset = HazeDataset(root, rows['test'], 'test')
    expected = tensor(Image.open(root / rows['test'][0]['input_relative']).resize((256, 256), Image.Resampling.BILINEAR))
    assert torch.equal(dataset[0]['input'], expected)
    assert HazeDataset(root, rows['test'], 'test', resolution='native')[0]['input'].shape == (3, 9, 7)


def test_exact_factories_initialization_shapes_backward():
    from src.shared.uicf_models import UICFPreBackbone
    from src.shared.color_query_unet import PlainUNetColorQuery
    full, wo = assert_initial_backbone_equivalence()
    assert isinstance(full, UICFPreBackbone) and isinstance(wo, PlainUNetColorQuery)
    np_full = sum(p.numel() for p in full.parameters())
    np_wo = sum(p.numel() for p in wo.parameters())
    assert np_full > np_wo
    assert np_full - np_wo == sum(p.numel() for p in full.uicf.parameters())
    for model in (full, wo):
        model.train()
        x = torch.rand(2, 3, 17, 19)
        pred = model(x)
        assert pred.shape == x.shape
        loss = CharbonnierLoss()(pred, torch.rand_like(pred))
        loss.backward()
        assert torch.isfinite(loss)
        assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)
        model.zero_grad(set_to_none=True)
    bad = canonical_model_config()
    bad['base_channels'] = 4
    with pytest.raises(ValueError, match='exactly equal'):
        build_model('full', bad)


def test_metric_mean_and_cluster_bootstrap():
    x = torch.zeros(2, 3, 16, 16)
    y = x.clone()
    y[0] = 0.1
    y[1] = 0.5
    ps, ss = batch_metrics(x, y, METRICS)
    assert torch.equal(ps, per_image_psnr(x, y))
    assert torch.equal(ss, per_image_ssim(x, y, window_size=11, sigma=1.5))
    assert not np.isclose(ps.mean().item(), 10 * np.log10(1 / ((x-y)**2).mean().item()))
    b = bootstrap([1, 1, 2], ['a','a','b'], iterations=2000)
    assert b == bootstrap([1, 1, 2], ['a','a','b'], iterations=2000)
    assert b['mean'] == pytest.approx(4/3) and b['clean_groups'] == 2
    assert b['ci95'] == [1.0, 2.0]
    with pytest.raises(ValueError):
        bootstrap([1], ['a'], iterations=100)
    assert bootstrap([3,3], ['a','b'], iterations=2000)['ci95'] == [3,3]


def test_backward_normalization_and_fp16_overflow():
    # A fixed FP16 intermediate reproduces the extra batch-factor overflow without CUDA.
    old = torch.tensor(0.25, dtype=torch.float16, requires_grad=True)
    (old.float() * 0.125 * 16 * 65536).backward()
    assert not torch.isfinite(old.grad)
    corrected = torch.tensor(0.25, dtype=torch.float16, requires_grad=True)
    (normalized_backward_loss(corrected.float() * 0.125, 16, 16) * 65536).backward()
    assert corrected.grad.item() == 8192 and torch.isfinite(corrected.grad)
    # Variable tail batches and truncated smoke groups equal each full group's mean gradient.
    x = torch.arange(1, 12, dtype=torch.float64)
    for acc, limit in ((1, 3), (2, 3), (3, 3), (4, 3), (3, 2)):
        accumulated = torch.tensor(0.2, dtype=torch.float64, requires_grad=True)
        reference = accumulated.detach().clone().requires_grad_()
        for step in range(1, limit + 1):
            batch = x[(step-1)*4:step*4]
            count = accumulation_sample_count(len(x), 4, acc, step, limit)
            loss = (accumulated * batch).square().mean()
            normalized_backward_loss(loss, len(batch), count).backward()
            if step % acc == 0 or step == limit:
                first = ((step-1)//acc)*acc*4
                (reference * x[first:step*4]).square().mean().backward()
                assert torch.allclose(accumulated.grad, reference.grad, atol=1e-12, rtol=1e-12)
                accumulated.grad = reference.grad = None


def test_amp_configuration_is_explicit():
    c = load_yaml(ROOT / 'configs/haze4k/full_v16.yaml')
    for value in (0, -1, float('inf'), float('nan')):
        c['training']['amp_init_scale'] = value
        with pytest.raises(ValueError, match='finite and positive'):
            validate_config(c)
    with pytest.raises(SystemExit):
        train(['--config', str(ROOT / 'configs/haze4k/full_v16.yaml'), '--fp32-smoke'])


def test_scale_ceiling_handles_real_scaler_growth_and_restore():
    # Public GradScaler APIs can be exercised on CPU without claiming CUDA validation.
    weight = torch.tensor(0.2, requires_grad=True)
    optimizer = torch.optim.SGD([weight], lr=0.01)
    scaler = torch.amp.GradScaler('cpu', init_scale=1024, growth_interval=2)
    events = 0
    for i in range(31):
        optimizer.zero_grad(set_to_none=True)
        scaler.scale(weight.square()).backward()
        scaler.unscale_(optimizer)
        assert torch.isfinite(weight.grad)
        scaler.step(optimizer)
        scaler.update()
        events += int(cap_scaler(scaler, 1024))
        assert scaler.get_scale() <= 1024
        if i == 14:
            state = scaler.state_dict()
            restored = torch.amp.GradScaler('cpu', init_scale=1024, growth_interval=2)
            restored.load_state_dict(state)
            assert not cap_scaler(restored, 1024)
            assert restored.state_dict() == state  # Ordinary resume must not reset its growth tracker.
            scaler = restored
    assert events == 15
    legacy = {'scale': 262144.0, 'growth_factor': 2., 'backoff_factor': 0.5,
              'growth_interval': 2000, '_growth_tracker': 448}
    capped = cap_scale_state(legacy, 1024)
    assert legacy['scale'] == 262144 and legacy['_growth_tracker'] == 448
    assert capped['scale'] == 1024 and capped['_growth_tracker'] == 0
    assert cap_scale_state({}, 1024) == {}
    with pytest.raises(ValueError, match='<='):
        validate_scale_policy(65536, 1024)
    with pytest.raises(ValueError):
        validate_scale_policy(1024, float('inf'))


@pytest.mark.parametrize('bounded', [False, True])
def test_default_growth_interval_can_overflow_later_even_after_successful_early_steps(bounded):
    # Real default 2000-update growth cadence and an actual FP16 backward intermediate.
    # This is a synthetic numerical regression, not a claim about the user's CUDA model.
    weight = torch.tensor(0.2, requires_grad=True)
    optimizer = torch.optim.SGD([weight], lr=0.)
    scaler = torch.amp.GradScaler('cpu', init_scale=1024 if bounded else 65536)
    failure = None
    for step in range(1, 4501):
        optimizer.zero_grad(set_to_none=True)
        loss = weight.to(torch.float16).float() * 0.25
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        if not torch.isfinite(weight.grad):
            failure = step
            break  # Same fail-fast behavior: no step/update after a nonfinite gradient.
        scaler.step(optimizer)
        scaler.update()
        cap_scaler(scaler, 1024 if bounded else None)
    if bounded:
        assert failure is None and step == 4500 and scaler.get_scale() == 1024
    else:
        assert failure == 4001 and scaler.get_scale() == 262144


def test_recovery_rejects_unreviewed_model_or_source_changes():
    current = source_hashes()
    old = {k: v for k, v in current.items() if not k.endswith(('precision.py', 'recover.py'))}
    old['src/haze4k_transfer/train.py'] = LEGACY_TRAIN_SHA256
    old['src/haze4k_transfer/prepare.py'] = LEGACY_PREPARE_SHA256
    differences = validate_recovery_sources(old, current)
    assert 'src/haze4k_transfer/train.py' in differences
    bad = dict(old)
    bad['src/shared/uicf_inr.py'] = 'unreviewed'
    with pytest.raises(ValueError, match='unreviewed source'):
        validate_recovery_sources(bad, current)
    bad = dict(old)
    bad['src/haze4k_transfer/train.py'] = 'unreviewed'
    with pytest.raises(ValueError, match='unreviewed source'):
        validate_recovery_sources(bad, current)


def test_nonfinite_gradients_still_fail_without_update(tmp_path, monkeypatch):
    import importlib
    module = importlib.import_module('src.haze4k_transfer.train')
    factory = module.build_model
    root = make_data(tmp_path / 'data', n=3, nt=2)
    split = tmp_path / 'split'
    prep.prepare(root, split, synthetic=True)
    captured = {}
    def build_with_fault(*args):
        model = factory(*args)
        weight = model.backbone.output_conv.weight
        captured.update(weight=weight, initial=weight.detach().clone())
        weight.register_hook(lambda gradient: torch.full_like(gradient, float('inf')))
        return model
    monkeypatch.setattr(module, 'build_model', build_with_fault)
    base = ROOT / 'experiments/haze4k_transfer'
    before = set(base.iterdir()) if base.exists() else set()
    try:
        with pytest.raises(FloatingPointError, match='no optimizer update applied'):
            train(['--config', str(ROOT / 'configs/haze4k/full_v16.yaml'), '--data-root', str(root),
                   '--split-dir', str(split), '--gpu', 'cpu', '--synthetic-smoke', '--smoke-epochs', '1',
                   '--amp-init-scale', '1024'])
        created = set(base.iterdir()) - before
        assert len(created) == 1
        run = created.pop()
        diagnostic = json.loads((run / 'nonfinite_diagnostic.json').read_text())
        assert diagnostic['step'] == 1 and diagnostic['inf_count'] > 0
        assert diagnostic['parameter'] == 'backbone.output_conv.weight'
        assert json.loads((run / 'status.json').read_text())['status'] == 'failed'
        assert not list((run / 'checkpoint').iterdir())
        assert torch.equal(captured['initial'], captured['weight'])
        assert load_yaml(run / 'config_resolved.yaml')['training']['amp_init_scale'] == 1024
    finally:
        for run in set(base.iterdir()) - before:
            shutil.rmtree(run)


def test_synthetic_train_resume_eval_compare(tmp_path):
    root = make_data(tmp_path / 'data', n=3, nt=2)
    split = tmp_path / 'split'
    prep.prepare(root, split, synthetic=True)
    runs = []
    wo_config = None
    try:
        for variant, file in [('full', 'full_v16'), ('wo_uicf', 'wo_uicf_v15_backbone')]:
            c = load_yaml(ROOT / f'configs/haze4k/{file}.yaml')
            c['training']['deterministic'] = True
            config = tmp_path / f'{file}.yaml'
            save_yaml(config, c)
            if variant == 'wo_uicf':
                wo_config = config
            run = train(['--config', str(config), '--data-root', str(root), '--split-dir', str(split),
                         '--gpu', 'cpu', '--smoke-epochs', '2', '--synthetic-smoke', '--stop-after-epoch', '1'])
            runs.append(run)
            before = torch_load(run / 'checkpoint/last.pt')
            assert before['epoch'] == 1 and before['optimizer']['state']
            assert 'amp_scaler' in before and 'loader_generator' in before and 'rng' in before
            if variant == 'full':
                # Build a recovery from a simulated pre-ceiling, recognized historical source version.
                # Remove only telemetry absent in the historical format, preserving actual trained state.
                for item in before['history']:
                    item.pop('amp_scale', None)
                    item.pop('amp_max_scale', None)
                before['source_hashes'] = {k: v for k, v in before['source_hashes'].items()
                                           if not k.endswith(('precision.py', 'recover.py'))}
                before['source_hashes']['src/haze4k_transfer/train.py'] = LEGACY_TRAIN_SHA256
                before['source_hashes']['src/haze4k_transfer/prepare.py'] = LEGACY_PREPARE_SHA256
                # Keep the source run for ordinary-resume tests intact by making a separate fixture run.
                import uuid
                historical = run.parent / f'full_legacy_fixture_{uuid.uuid4().hex}'
                shutil.copytree(run, historical, ignore=shutil.ignore_patterns('checkpoint'))
                (historical / 'checkpoint').mkdir()
                torch.save(before, historical / 'checkpoint/last.pt')
                torch.save(before, historical / 'checkpoint/best_psnr.pt')
                runs.append(historical)
                prior = {p.relative_to(historical).as_posix(): prep.sha256(p) for p in historical.rglob('*') if p.is_file()}
                branch = recover(['--checkpoint', str(historical / 'checkpoint/last.pt'), '--amp-max-scale', '1024'])
                runs.append(branch)
                assert prior == {p.relative_to(historical).as_posix(): prep.sha256(p) for p in historical.rglob('*') if p.is_file()}
                branch_cp = torch_load(branch / 'checkpoint/last.pt')
                assert branch_cp['epoch'] == 1 and branch_cp['config']['smoke']
                assert all(torch.equal(value, branch_cp['model'][name]) for name, value in before['model'].items())
                assert branch_cp['rng']['python'] == before['rng']['python']
                assert torch.equal(branch_cp['rng']['torch'], before['rng']['torch'])
                assert torch.equal(branch_cp['loader_generator'], before['loader_generator'])
                del branch_cp
                train(['--resume', str(branch / 'checkpoint/last.pt'), '--gpu', 'cpu'])
                branch_cp = torch_load(branch / 'checkpoint/last.pt')
                assert branch_cp['epoch'] == 2 and branch_cp['config']['training']['amp_max_scale'] == 1024
                assert len(branch_cp['history']) == 2
                assert all(int(v['step']) == 2 for v in branch_cp['optimizer']['state'].values())
                del branch_cp
                evaluate(['--run-dir', str(branch), '--gpu', 'cpu', '--resolution', '256'])
                recovered_summary = json.loads((branch / 'test_256/summary.json').read_text())
                assert recovered_summary['smoke']  # Mixed-policy recovery cannot masquerade as a main run.
            with pytest.raises(ValueError, match='completed target'):
                evaluate(['--run-dir', str(run), '--gpu', 'cpu'])
            # Resume really restores optimizer/RNG, then completes the second epoch.
            del before
            train(['--resume', str(run / 'checkpoint/last.pt'), '--gpu', 'cpu'])
            after = torch_load(run / 'checkpoint/last.pt')
            assert after['epoch'] == 2 and len(after['history']) == 2
            assert all(int(v['step']) == 2 for v in after['optimizer']['state'].values())
            del after
            evaluate(['--run-dir', str(run), '--gpu', 'cpu', '--resolution', '256'])
        # Same CPU/device/workers deterministic epoch-boundary resume must match continuous training exactly.
        uninterrupted = train(['--config', str(wo_config), '--data-root', str(root), '--split-dir', str(split),
                               '--gpu', 'cpu', '--smoke-epochs', '2', '--synthetic-smoke'])
        runs.append(uninterrupted)
        continuous = torch_load(uninterrupted / 'checkpoint/last.pt')
        wo_run = next(r for r in runs if r.name.startswith('wo_uicf_'))
        resumed = torch_load(wo_run / 'checkpoint/last.pt')
        assert continuous['model'].keys() == resumed['model'].keys()
        assert all(torch.equal(v, resumed['model'][k]) for k, v in continuous['model'].items())
        assert torch.equal(continuous['loader_generator'], resumed['loader_generator'])
        assert torch.equal(continuous['rng']['torch'], resumed['rng']['torch'])
        del continuous, resumed
        directory = compare(['--full-run', str(runs[0]), '--wo-run', str(wo_run), '--bootstrap-iterations', '2000'])
        assert (directory / 'paper_table.md').is_file()
        assert 'SYNTHETIC' in (directory / 'paper_table.md').read_text()
        assert len(list((directory / 'panels').glob('*.png'))) == 2
        fd, fs, fr = load_result(runs[0], '256')
        wd, ws, wr = load_result(wo_run, '256')
        bad = copy.deepcopy(wr)
        bad.pop(next(iter(bad)))
        with pytest.raises(ValueError, match='IDs differ'):
            align(fs, ws, fr, bad, '256')
        bad = copy.deepcopy(ws)
        bad['protocol'] = protocol('native')
        with pytest.raises(ValueError, match='protocol'):
            align(fs, bad, fr, wr, '256')
        # Native protocol must preserve odd-sized image dimensions, no silent resize/tile.
        evaluate(['--run-dir', str(runs[0]), '--gpu', 'cpu', '--resolution', 'native'])
        bad = copy.deepcopy(ws)
        bad['checkpoint'] = 'last'
        with pytest.raises(ValueError, match='checkpoint'):
            align(fs, bad, fr, wr, '256')
        # Preserve small evidence, never retain large temporary checkpoints in repository.
        evidence = tmp_path / 'evidence'
        shutil.copytree(directory, evidence)
        print('Synthetic pipeline complete; checkpoint, optimizer, RNG restore and 256/native eval verified.')
    finally:
        for run in runs:
            shutil.rmtree(run)


@pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA unavailable: enabled AMP/scaler restore must run on AutoDL')
def test_cuda_enabled_amp_resume(tmp_path):
    root = make_data(tmp_path / 'data', n=3, nt=2)
    split = tmp_path / 'split'
    prep.prepare(root, split, synthetic=True)
    run = None
    try:
        run = train(['--config', str(ROOT / 'configs/haze4k/full_v16.yaml'), '--data-root', str(root),
                     '--split-dir', str(split), '--gpu', '0', '--smoke-epochs', '2', '--synthetic-smoke',
                     '--stop-after-epoch', '1'])
        first = torch_load(run / 'checkpoint/last.pt')
        assert first['amp_scaler']['scale'] > 0 and first['rng']['cuda']
        tracker = first['amp_scaler']['_growth_tracker']
        del first
        train(['--resume', str(run / 'checkpoint/last.pt'), '--gpu', '0'])
        last = torch_load(run / 'checkpoint/last.pt')
        assert last['epoch'] == 2
        assert last['amp_scaler']['_growth_tracker'] == tracker + 1
    finally:
        if run:
            shutil.rmtree(run)
